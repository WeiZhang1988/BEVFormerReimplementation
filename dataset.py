import torch
import numpy as np
import os
import glob
import contextlib
import collections
from PIL import ExifTags, Image, ImageOps
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from tqdm import tqdm


for orientation in ExifTags.TAGS.keys():
  if ExifTags.TAGS[orientation] == 'Orientation':
    break

def exif_size(img):
  # Returns exif-corrected PIL size
  s = img.size  # (width, height)
  with contextlib.suppress(Exception):
    rotation = dict(img._getexif().items())[orientation]
    if rotation in [6, 8]:  # rotation 270 or 90
      s = (s[1], s[0])
  return s

def preprocess_images_list_and_labels_list(images_list, labels_list):
  images_dic = {}
  labels_dic = {}
  for images_item in images_list:
    image_body = images_item.split('.')[-2]
    image_frame_ID = image_body.split('_')[-2]
    if image_frame_ID in images_dic:
      images_dic[image_frame_ID].append(images_item)
    else:
      images_dic[image_frame_ID] = [images_item]
  for images_item, lable_item in zip(images_dic, labels_list):
    images_dic[images_item].sort()
    label_body = lable_item.split('.')[-2]
    label_frame_ID = label_body.split('_')[0]
    labels_dic[label_frame_ID] = lable_item
  images_od = collections.OrderedDict(sorted(images_dic.items()))
  labels_od = collections.OrderedDict(sorted(labels_dic.items()))
  for image_frame_ID, label_frame_ID in zip(images_od,labels_od):
    assert image_frame_ID == image_frame_ID
  return list(images_od.values()), list(labels_od.values())

def xyxy2xywh(x):
  # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
  y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
  y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
  y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
  y[..., 2] = x[..., 2] - x[..., 0]  # width
  y[..., 3] = x[..., 3] - x[..., 1]  # height
  return y

def segments2boxes(segments):
  # Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
  boxes = []
  for s in segments:
    x, y = s.T  # segment xy
    boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
  return xyxy2xywh(np.array(boxes))  # cls, xywh

def verify_image_label(args):
  # Verify one (6images-label) pair
  segments = []  # number (missing, found, empty, corrupt), message, segments
  im_files, lb_file = args
  try:
    # verify images
    ims = [Image.open(im_file) for im_file in im_files]
    for im in ims:
      im.verify()
    shapes = [exif_size(im) for im in ims]  # image size
    # verify labels
    if os.path.isfile(lb_file):
      with open(lb_file) as f:
        lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
        if any(len(x) > 6 for x in lb):  # is segment
          classes = np.array([x[0] for x in lb], dtype=np.float32)
          segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
          lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
        lb = np.array(lb, dtype=np.float32)
      nl = len(lb)
      if nl:
        assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
        assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
        assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
        _, i = np.unique(lb, axis=0, return_index=True)
        if len(i) < nl:  # duplicate row check
          lb = lb[i]  # remove duplicates
          if segments:
            segments = [segments[x] for x in i]
      else:
        lb = np.zeros((0, 5), dtype=np.float32)
    else:
      lb = np.zeros((0, 5), dtype=np.float32)
    return im_files[0], im_files, lb, shapes, segments
  except Exception:
    print("exception")
    return [None, None, None, None, None]

class BEVDataset(torch.utils.data.Dataset):
  """
  suppose file names follow the pattern:
  images: frameID_camID.png
  labels: frameID.txt
  """
  def __init__(self, img_dir='./data/images', label_dir='./data/labels', cache_dir='./data/cache', lidar2img_trans=torch.tile(torch.eye(4),(6,1,1)), num_levels=2, batch_size=16, num_threads=1):
    """
    Args:
      img_dir         (string):  The path to images
        Default: './data/images'
      label_dir       (string):  The path to labels
        Default: './data/labels'
      cache_dir       (string):  The path to cache
        Default: './data/cache'
      lidar2img_trans (Tensor [num_cam, 4, 4]): The tranformation matrices from BEV to limages
        Default: Tensor of [eye(4), ..., eye(4)], 6 X eye(4) in total
      num_levels:     (int):     The number of levels
        Default: 2
      batch_size      (int):     The batch size
        Default: 16
      num_threads     (int):     The number of threads
        Default: 1
    """
    super().__init__()
    self.num_levels      = num_levels
    self.lidar2img_trans = lidar2img_trans
    self.num_threads     = num_threads
    self.img_dir       = img_dir
    self.label_dir     = label_dir
    self.cache_dir     = cache_dir
    self.images_png = glob.glob(img_dir+'/*')
    self.labels_txt = glob.glob(label_dir+'/*')
    assert os.path.exists(img_dir),   "image directory does not exist"
    assert os.path.exists(label_dir), "label directory does not exist"
    assert self.images_png, f'No images found'
    assert self.labels_txt, f'No labels found'
    assert len(self.images_png) == len(self.labels_txt) * self.lidar2img_trans.shape[0]
    self.im_files, self.label_files = preprocess_images_list_and_labels_list(self.images_png,self.labels_txt)
    cache_path = cache_dir + '.cache'
    try:
      cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
    except Exception:
      cache, exists = self.cache_labels(cache_path), False  # run cache ops
    images, labels, shapes, self.segments = zip(*cache.values())
    nl = len(np.concatenate(labels, 0))  # number of labels
    assert nl > 0, f'All labels empty in {cache_path}, can not start training.'
    self.labels = list(labels)
    self.shapes = [np.array(shape) for shape in shapes]
    # Create indices
    n = len(self.shapes)  # number of image groups
    bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
    nb = bi[-1] + 1  # number of batches
    self.batch = bi  # batch index of image
    self.n = n
    self.indices = range(n)
    # Update labels
    include_class = []  # filter labels to include only these classes (optional)
    self.segments = list(self.segments)
    include_class_array = np.array(include_class).reshape(1, -1)
    for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
      if include_class:
        j = (label[:, 0:1] == include_class_array).any(1)
        self.labels[i] = label[j]
        if segment:
          self.segments[i] = [segment[idx] for idx, elem in enumerate(j) if elem]

  def __getitem__(self, index):
    pass

  def load_image(self, i):
    # im_files list of image file names of ith frame
    im_files = self.im_files[i]
    ims = [Image.open(im_file) for im_file in im_files]
    lidar2img = self.lidar2img_trans
    return ims, lidar2img

  def cache_labels(self, path=Path('./data/labels.cache')):
    # Cache dataset labels, check images and read shapes
    x = {}  # dict
    with Pool(self.num_threads) as pool:
      pbar = tqdm(pool.imap(verify_image_label, zip(self.im_files, self.label_files)),total=len(self.im_files))
      for first_im_file, im_file, lb, shape, segments in pbar:
        if im_file:
          x[first_im_file] = [im_file, lb, shape, segments]
      pbar.close()
    try:
        np.save(path, x)  # save cache for next time
        path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
    except Exception:
        pass
    return x
    
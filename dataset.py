import torch
import os
import pandas as pd
from PIL import ExifTags, Image, ImageOps
import contextlib

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

class BEVDataset(torch.utils.data.Dataset):
  """
  suppose file names follow the pattern:
  images: frameID_camID.png
  labels: frameID.txt
  """
  def __init__(self, img_dir, label_dir):
    super().__init__()
    self.img_dir   = img_dir
    self.label_dir = label_dir
    self.images = os.listdir(img_dir)
    self.labels = os.listdir(label_dir)
    assert os.path.exists(img_dir),   "image directory does not exist"
    assert os.path.exists(label_dir), "label directory does not exist"
    assert self.images, f'No images found'
    assert self.labels, f'No labels found'
    
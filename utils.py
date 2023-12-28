from PIL import Image, ImageFilter
import numpy as np
import torch
import os

def save_checkpoint(state, checkpoint='./model/model.pth.tar'):
  print("--> saving checkpoint")
  torch.save(state, checkpoint)


def load_checkpoint(checkpoint, model, optimizer):
  print("--> loading checkpoint")
  model.load_state_dict(checkpoint["state_dict"])
  optimizer.load_state_dict(checkpoint["optimizer"])

class CarlaInstanceSemeantic2CocoLabelConverter:
  """
  Args:
    input_path    (string): The path of input files
      Default: "./data/instance_semantics/"
    output_path   (string): The path to output files
      Default: "./data/labels/"
    resize_to     (tuple):  The size to resize to
      Default: None
  Notes:
    Carla instance semantics raw data is BGRA data with R channel representing tags, B and G channels combined representing ID
  """
  def __init__(self,input_path="./data/instance_semantics/",output_path="./data/labels/",resize_to=None):
    self.input_path = input_path
    self.output_path = output_path
    self.resize_to = resize_to
    assert os.path.exists(input_path), "input path does not exist"
    if not os.path.exists(output_path):
      os.makedirs(output_path)
  def convert(self):
    self.process_one_folder(self.input_path, self.output_path, self.resize_to)
  def process_one_folder(self,input_path,output_path,resize_to):
    file_list = os.listdir(input_path)
    for ins in file_list:
      body   = ins.split('.')[0]
      input_file_name  = input_path+ins
      output_file_name = output_path+body+'.txt'
      tag_label = self.process_one_image(input_file_name,resize_to)
      with open(output_file_name,'w') as filehandle:
        for listitem in tag_label:
          filehandle.write((f'{listitem}\n'.replace('[','').replace(']','').replace(',','')))
  def process_one_image(self,input_file_name,resize_to):
    img = Image.open(input_file_name)
    if resize_to:
      img = img.resize(resize_to, Image.NEAREST)
    H, W = img.size
    numpy_array = np.array(img,dtype=np.uint8)
    channel0 = numpy_array[...,0]
    channel1 = numpy_array[...,1]
    channel2 = numpy_array[...,2]
    stacked  = np.stack((channel0,channel1),axis=-1).astype(np.uint64)
    combined = np.zeros_like(channel0,dtype=np.uint64)
    for i, x in enumerate(stacked):
      for j, y in enumerate(x):
        combined[i,j] = np.uint64(''.join([str(item) for item in y]))
    tag_set = set(channel2.flatten().tolist())
    tag_label = []
    for tag in tag_set:
      im = Image.fromarray(channel2==np.uint8(tag)).filter(ImageFilter.FIND_EDGES)
      edge = np.array(im,dtype=np.uint64)
      combined_edge = edge * combined
      ID_set = set(combined_edge.flatten().tolist())
      for id in ID_set:
        if id >0:
          indices = np.where(combined_edge==np.uint64(id))
          if len(indices[0])>3:
            assert len(indices[0]) == len(indices[1]), "x y indices must have same length"
            indices_x_flt = indices[0].astype(np.float128) / np.float128(H)
            indices_y_flt = indices[1].astype(np.float128) / np.float128(W)
            indices = np.array([indices_x_flt,indices_y_flt]).flatten('F')
            tag_list = [tag] + (indices.tolist())
            tag_label.append(tag_list)
    return tag_label

if __name__ == "__main__":
  converter = CarlaInstanceSemeantic2CocoLabelConverter(input_path="./data/instance_semantics/",output_path="./data/labels/",resize_to=(96,96))
  converter.convert()
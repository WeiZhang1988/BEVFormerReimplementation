from PIL import Image, ImageFilter
import numpy as np
import os

class CarlaInstanceSemeantic2CocoLabelConverter:
  def __init__(self,input_path="./INS/",output_path="./PROCESSED/",to_size=None):
    self.input_path = input_path
    self.output_path = output_path
    self.to_size = to_size
  def convert(self):
    self.process_one_folder(self.input_path, self.output_path, self.to_size)
  def process_one_folder(self,input_path,output_path,to_size):
    dir_list = os.listdir(input_path)
    for ins in dir_list:
      body   = ins.split('.')[0]
      input_file_name  = input_path+ins
      output_file_name = output_path+body+'.txt'
      tag_label = self.process_one_image(input_file_name,to_size)
      with open(output_file_name,'w') as filehandle:
        for listitem in tag_label:
          filehandle.write((f'{listitem}\n'.replace('[','').replace(']','')))
  def process_one_image(self,input_file_name,to_size):
    img = Image.open(input_file_name)
    if to_size:
      img = img.resize(to_size)
    numpy_array = np.array(img,dtype=np.uint8)
    channel0 = numpy_array[...,0]
    channel1 = numpy_array[...,1]
    channel2 = numpy_array[...,2]
    stacked  = np.stack((channel0,channel1),axis=-1,dtype=np.uint64)
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
          indices = np.array([indices[0],indices[1]]).flatten('F')
          tag_list = [tag] + (indices.tolist())
          tag_label.append(tag_list)
    return tag_label




if __name__ == "__main__":
  converter = CarlaInstanceSemeantic2CocoLabelConverter(input_path="./INS/",output_path="./PROCESSED/",to_size=(96,96))
  converter.convert()
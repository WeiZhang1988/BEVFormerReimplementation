import torch
import os
import pandas as pd
from PIL import Image

class BEVDataset(torch.utils.data.Dataset):
  def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None):
    super().__init__()
    
import os
import time
import torch
import torch.optim as optim

from bevformer import BEVFormer
from dataset import BEVDataset
#from loss import Loss
from tqdm import tqdm

torch.manual_seed(123)
learning_rate = 2e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size  = 32
num_epochs  = 300
num_threads = 4
img_dir   = "./data/images"
label_dir = "./data/labels"
cache_dir = "./data/cache"

def train_fn(train_loader, model, optimizer, loss_fn):
  loop = tqdm(train_loader, leave=True)
  mean_loss = []
  for 


def main():
  pass

if __name__ == "__main__":
  main()
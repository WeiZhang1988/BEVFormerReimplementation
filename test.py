import torch
from torch import nn
import torch.nn.functional as F
from utils import load_checkpoint, save_checkpoint
from backbone import BackBone
from encoder import EncoderLayer, Encoder
from heads import Segment
from decoder import DecoderLayer, Decoder
from bevformer import BEVFormer
from dataset import BEVDataset
from dataloader import InfiniteDataLoader
from loss import BEVLoss
from tqdm import tqdm
import config


class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.cov = nn.Conv2d(3,5,kernel_size=3,stride=3)
  def forward(self,x):
    print(f"1 allocated cuda {torch.cuda.memory_allocated()}")
    x = self.cov(x)
    print(f"2 allocated cuda {torch.cuda.memory_allocated()}")
    return x.sum()


train_dataset = BEVDataset(img_dir=config.data_img_dir, 
                             label_dir=config.data_label_dir, 
                             cache_dir=config.data_cache_dir, 
                             lidar2img_trans=config.data_lidar2image_trans, 
                             num_levels=config.data_num_levels,
                             batch_size=config.data_batch_size, 
                             num_threads=config.data_num_threads, 
                             bev_size=config.data_bev_size, 
                             overlap=config.data_overlap)
train_dataloader = InfiniteDataLoader(dataset=train_dataset,
                                      batch_size=config.data_batch_size,
                                      num_workers=config.data_num_threads,
                                      pin_memory=config.data_pin_memory,
                                      collate_fn=BEVDataset.collate_fn,
                                      shuffle=config.data_shuffle,
                                      drop_last=config.data_drop_last,)

model = Model().to(config.device)
for i_epoch in range(100):
    loop = tqdm(train_dataloader, leave=True)
    for batch_idx, (imgs_outs, lidar2img_transes, labels_outs, masks_outs) in enumerate(loop):
      print(f"<--- {i_epoch} th epoch, {batch_idx} th batch --->")
      #print(f"1 allocated cuda {torch.cuda.memory_allocated()}")
      imgs_outs_, lidar2img_transes_, labels_outs_, masks_outs_ = imgs_outs.to(config.device).to(non_blocking=True), lidar2img_transes.to(config.device).to(non_blocking=True), labels_outs.to(config.device).to(non_blocking=True), masks_outs.to(config.device).to(non_blocking=True)
      out = model(imgs_outs_[0])
      out.backward(retain_graph=True)
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from einops import rearrange, einsum

from attentions import *
from encoder import *
from decoder import *

class Algorithm:
  def __init__(self,bevformer=None,lr=1e-4,cls_weight=20.0,bbx_weight=0.25):
    self.bevformer  = bevformer
    self.cls_weight = cls_weight
    self.bbx_weight = bbx_weight
    self.optimizer  = Adam(self.bevformer.parameters(), lr=lr)
    self.loss       = torch.tensor(0.0)
    self.output_cls = torch.tensor(0.0)
    self.output_bbx = torch.tensor(0.0)
  def predict(self,inputs):
    self.output_cls, self.output_bbx = self.bevformer(inputs)
    return self.output_cls, self.output_bbx
  def update(self,ground_truth):
    loss_bbx  = F.smooth_l1_loss(self.output_bbx, ground_truth.bbx, size_average=True)
    loss_cls  = self.softmax_focal_loss(self.output_cls, ground_truth.cls)
    self.loss = self.bbx_weight * loss_bbx + self.cls_weight * loss_cls
    self.optimizer.zero_grad()
    self.loss.backward(retain_graph=False)
    self.optimizer.step()
  def softmax_focal_loss(self,inputs,targets,reduction='mean',alpha=1,gamma=2):
    log_probs = F.log_softmax(inputs, dim=-1)
    probs = torch.exp(log_probs)
    targets = nn.functional.one_hot(targets, num_classes=inputs.size(-1)).float()
    alpha_t = (1 - probs) ** gamma * alpha + targets * probs
    loss = -torch.sum(alpha_t * log_probs, dim=-1)
    if reduction == 'mean':
      return loss.mean()
    elif reduction == 'sum':
      return loss.sum()
    else:
      return loss


class BEVFormer(nn.Module):
  count = 0
  def __init__(self,encoder=None,decoder=None,lr=1e-4,device=torch.device("cpu")):
    super().__init__()
    BEVFormer.count+=1
    self.encoder = encoder
    self.decoder = decoder
  def __del__(self):
    BEVFormer.count-=1	
  def forward(self,inputs):
    print(f"bevformer-1 allocated cuda {torch.cuda.memory_allocated()}")
    tmp_out = self.encoder(inputs['list_leveled_images'],inputs['spat_lidar2img_trans'])
    print(f"bevformer-2 allocated cuda {torch.cuda.memory_allocated()}")
    cls, crd, segments, proto = self.decoder(tmp_out[-1])
    #print(f"-3 allocated cuda {torch.cuda.memory_allocated()}")
    del tmp_out
    torch.cuda.empty_cache()
    #print(f"-4 allocated cuda {torch.cuda.memory_allocated()}")
    return cls, crd, segments, proto


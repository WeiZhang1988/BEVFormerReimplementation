import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, einsum

from attentions import *

class Decoder(nn.Module):
  def __init__(self,backbone=None,bevformerlayer=None,device=torch.device("cpu")):
    super().__init__()
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, einsum

from attentions import *


class Encoder(nn.Module):
  pass

class BEVFormerLayer(nn.Module):
  """
  Args:
    num_bev_queue (int):   The number of BEVs to be used
      Default: 2
    num_cams      (int):   The number of cameras
      Default: 6
    dropout       (float): The drop out rate
      Default: 0.1
    deformable_attention (nn): as name suggests
    -----MultiScaleDeformableAttention3D arguments-----
    embed_dims (int): The embedding dimension of attention. The same as inputs embedding dimension.
      Default: 256
    num_heads  (int): The number of heads.
      Default: 8
    num_levels (int): The number of scale levels
      Default: 4
    num_points (int): The number of sampling points
      Default: 4
    -----Device-----
    device (torch.device): The device
      Default: cpu
  """
  def __init__(self,num_bev_queue=2,num_cams=6,dropout=0.1,embed_dims=256,num_heads=8,num_levels=4,num_points=4,device=torch.device("cpu")):
    super().__init__()
    self.num_bev_queue  = num_bev_queue
    self.num_cams       = num_cams
    self.dropout        = dropout
    self.embed_dims     = embed_dims
    self.num_heads      = num_heads
    self.num_levels     = num_levels
    self.num_points     = num_points
    self.device         = device
    self.NN_tempAttn    = TemporalSelfAttention(              dropout,embed_dims,num_heads,num_levels,num_points,num_bev_queue,device)
    self.NN_spatAttn    = SpatialCrossAttention(num_cams,     dropout,embed_dims,num_heads,num_levels,num_points,1,device)
    self.NN_addNorm1    = AddAndNormLayer(None,embed_dims,device=device)
    self.NN_addNorm2    = AddAndNormLayer(None,embed_dims,device=device)
    self.NN_addNorm3    = AddAndNormLayer(None,embed_dims,device=device)
    self.NN_ffn         = nn.Linear(embed_dims,embed_dims,device=device)
  def forward(self,spatAttn_key,spatAttn_value,tempAttn_query,tempAttn_key_hist=[],tempAttn_value_hist=[],reference_points=None,spatial_shapes=None,reference_points_cam=None,bev_mask=None):
    """
    Args:
      spatAttn_key         (Tensor [num_cams, bs, num_key, embed_dims]):     The spatial key
      spatAttn_value       (Tensor [num_cams, bs, num_value, embed_dims]):   The spatial value
      tempAttn_query       (Tensor [bs, num_query, embed_dims]):             The temporal query
      tempAttn_key_hist    (list of Tensor [bs, num_key,  embed_dims]s):     The temporal key history. num_key should equel to num_levels * num_points
      tempAttn_value_hist  (list of Tensor [bs, num_value,embed_dims]s):     The temporal value history
      reference_points     (Tensor [bs, num_query, num_levels, 2]):          The normalized reference points. Passed though to multi scale deformable attention layer
      spatial_shapes       (Tensor [num_levels, 2]):                         The spatial shape of features in different levels. Passed though to multi scale deformable attention layer
      reference_points_cam (Tensor [num_cam, bs, num_query, num_levels, 2]): The image pixel ratio projected from reference points to each camera
      bev_mask             (Tensor [num_cam, bs, num_query, num_levels]):    Which of reference_points_cam is valid
    Returns:
      currentBEV           (Tensor [bs, num_query, emded_dims])
    """
    tempAttn = self.NN_tempAttn(query=tempAttn_query,key_hist=tempAttn_key_hist,value_hist=tempAttn_value_hist,reference_points=reference_points,spatial_shapes=spatial_shapes)
    addNorm1 = self.NN_addNorm1(x=tempAttn,y=tempAttn_query)
    spatAttn = self.NN_spatAttn(query=addNorm1,key=spatAttn_key,value=spatAttn_value,reference_points=None,spatial_shapes=spatial_shapes,reference_points_cam=reference_points_cam,bev_mask=bev_mask)
    addNorm2 = self.NN_addNorm1(x=spatAttn,y=addNorm1)
    ffn      = self.NN_ffn(addNorm2)
    addNorm3 = self.NN_addNorm1(x=ffn,y=addNorm2)
    return addNorm3



class AddAndNormLayer(nn.Module):
  """
  Args:
    num_query (int): The number of query
      Default: None
    embed_dims(int): The embedding dimension
      Default: 6
    -----Device-----
    device (torch.device): The device
      Default: cpu
  -------------------------------------------------------------------------------------------------
  Note:
    1, Input of this layer is of shape [bs, num_query, embed_dims]
    2, This implementation can normalize last two dimensions, which are num_query and embed_dims, if num_query is provided.
       Otherwise, only the last dimension which is embed_dims is normalized
  """
  def __init__(self,num_query=None, embed_dims=256,device=torch.device("cpu")):
    super().__init__()
    self.num_query  = num_query
    self.embed_dims = embed_dims
    self.device     = device
    if num_query is not None:
      self.NN_layerNorm = nn.LayerNorm([num_query,embed_dims],device=device)
    else:
      self.NN_layerNorm = nn.LayerNorm(embed_dims,device=device)
  def forward(self,x,y):
    """
    Args:
      x       (Tensor [bs, num_query, emded_dims]): The tensor to be added
      y       (Tensor [bs, num_query, emded_dims]): The tensor to add
    Returns:
      addNorm (Tensor [bs, num_query, emded_dims])
    """
    added = x + y
    return self.NN_layerNorm(added)
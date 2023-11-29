import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, einsum

from attentions import *

class Decoder(nn.Module):
  def __init__(self,backbone=None,bevformerlayer=None,device=torch.device("cpu")):
    super().__init__()




class DecoderLayer(nn.Module):
  def __init__(self,full_num_query=100,full_dropout=0.1,full_embed_dims=256,full_num_heads=8,full_num_levels=1,full_num_points=4,\
                    query_H=200,query_W=200,custom_dropout=0.1,custom_embed_dims=256,custom_num_heads=8,custom_num_levels=1,custom_num_points=4,\
                    code_size=10,device=torch.device("cpu")):
    super().__init__()
    assert full_embed_dims == custom_embed_dims, "embed_dims for full and custom attention must be the same"
    self.full_num_query     = full_num_query
    self.full_dropout       = full_dropout
    self.full_embed_dims    = full_embed_dims
    self.full_num_heads     = full_num_heads
    self.full_num_levels    = full_num_levels
    self.full_num_points    = full_num_points
    self.query_H            = query_H
    self.query_W            = query_W
    self.custom_dropout     = custom_dropout
    self.custom_embed_dims  = custom_embed_dims
    self.custom_num_heads   = custom_num_heads
    self.custom_num_levels  = custom_num_levels
    self.custom_num_points  = custom_num_points
    self.code_size          = code_size
    self.device             = device
    embed_dims              = full_embed_dims
    self.embed_dims         = embed_dims
    self.spatial_shapes     = torch.Tensor([[query_H,query_W]]).to(device)
    # self.NNP_query_origin [1(extends to bs), num_query, embed_dims]
    self.NNP_query_origin  = nn.Parameter(torch.ones(1,full_num_query,full_embed_dims,device=device)*0.98)
    # self.NNP_query_pos [1(extends to bs), num_query, embed_dims]
    self.NNP_query_pos     = nn.Parameter(torch.ones(1,full_num_query,full_embed_dims,device=device)*0.95)
    # self.query [1(extends to bs), num_query, embed_dims]
    self.query             = self.NNP_query_origin + self.NNP_query_pos
    self.NN_ref_points     = nn.Linear(self.embed_dims, 3, device=device)
    # self.reference_points [1(extends to bs), num_query, 3]
    self.reference_points  = self.NN_ref_points(self.NNP_query_pos)
    self.NN_fullAttn       = FullAttention(full_dropout,full_embed_dims,full_num_heads,full_num_levels,full_num_points,device)
    self.NN_custAttn       = CustomAttention(custom_dropout,custom_embed_dims,custom_num_heads,custom_num_levels,custom_num_points,device)
    self.NN_addNorm1       = AddAndNormLayer(None,embed_dims,device=device)
    self.NN_addNorm2       = AddAndNormLayer(None,embed_dims,device=device)
    self.NN_addNorm3       = AddAndNormLayer(None,embed_dims,device=device)
    self.NN_ffn            = nn.Linear(embed_dims,embed_dims,device=device)
    self.NN_regFiner       = RegressionFiner(embed_dims,code_size,device=device)
  def forward(self,bev_feat):
    bs, num_query, emded_dims = bev_feat.shape
    # self.query [1,  num_query, embed_dims]
    # ---->query [bs, num_query, embed_dims]
    query = self.query.repeat(bs,1,1)
    # self.reference_points [1,  num_query, 3]
    # ---->reference_points [bs, num_query, 3]
    reference_points = self.reference_points.repeat(bs,1,1)
    output = query
    #<------------------main body
    reference_points_input = reference_points[..., :2].unsqueeze(2)
    fullAttn = self.NN_fullAttn(query=output,key=output,value=output)
    addNorm1 = self.NN_addNorm1(x=fullAttn,y=output)
    custAttn = self.NN_custAttn(query=addNorm1,key=bev_feat,value=bev_feat,reference_points=reference_points_input,spatial_shapes=self.spatial_shapes)
    addNorm2 = self.NN_addNorm2(x=custAttn,y=addNorm1)
    ffn      = self.NN_ffn(addNorm2)
    output   = self.NN_addNorm3(x=ffn,y=addNorm2)
    tmp = self.NN_regFiner(output)
    new_reference_points = torch.zeros_like(reference_points)
    new_reference_points[..., :2]  = tmp[..., :2]  + self.inverse_sigmoid(reference_points[..., :2])
    new_reference_points[..., 2:3] = tmp[..., 4:5] + self.inverse_sigmoid(reference_points[..., 2:3]) 
    new_reference_points = new_reference_points.sigmoid()
    reference_points = new_reference_points
    #--------------------------->
    return output
  def inverse_sigmoid(self,x,eps=1e-5):
    """Inverse function of sigmoid.
    Args:
      x   (Tensor): The tensor to do the inverse.
      eps (float):  EPS avoid numerical overflow. Defaults 1e-5.
    Returns:
      inversed sigmoid (Tensor)
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class RegressionFiner(nn.Module):
  def __init__(self,embed_dims=256,code_size=10,device=torch.device("cpu")):
    super().__init__()
    self.embed_dims = embed_dims
    self.code_size  = code_size
    self.device     = device
    reg_branch = []
    for _ in range(2):
      reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims).to(device))
      reg_branch.append(nn.ReLU().to(device))
    reg_branch.append(nn.Linear(self.embed_dims, code_size).to(device))
    self.reg_branch = nn.Sequential(*reg_branch).to(device)
  def forward(self,x):
    return self.reg_branch(x)
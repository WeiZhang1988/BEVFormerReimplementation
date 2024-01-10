import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, einsum

from attentions import *
from heads import *

class Decoder(nn.Module):
  def __init__(self,num_classes=10,point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],decoderlayer=None,segmenthead=None,device=torch.device("cpu")):
    """
    Args:
      num_classes    (int):           The number of classes to detect
        Default: 10
      point_cloud_range (Tensor [6]): The range of point cloud
        Default: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
      decoderlayer   (nn module):     The decoderlayer module
        Default: None
      -----Device-----
      device (torch.device): The device
        Default: cpu
    """
    super().__init__()
    self.num_classes       = num_classes
    self.point_cloud_range = point_cloud_range
    self.decoderlayer      = decoderlayer
    self.num_layers        = decoderlayer.num_layers
    self.embed_dims        = decoderlayer.embed_dims
    self.num_key           = decoderlayer.custom_num_levels * decoderlayer.custom_num_points
    self.num_value         = decoderlayer.custom_num_levels * decoderlayer.custom_num_points
    self.code_size         = decoderlayer.code_size
    self.segmenthead       = segmenthead
    # self.NNP_keyvalue_pos [1(extends to bs), num_levels * num_points, embed_dims]
    self.NNP_keyvalue_pos  = nn.Parameter(torch.ones(1,decoderlayer.custom_num_levels * decoderlayer.custom_num_points,self.embed_dims,device=device)*0.95)
    # in original code, for each layer there is an independent regression finer and classifer. For simplification purpose, here shares one finer and classifier for all layers
    self.NN_regFiner       = RegressionFiner(self.embed_dims,self.code_size,device=device)
    self.NN_classfier      = Classifier(self.embed_dims,num_classes,device=device)
  def forward(self,encoder_feat):
    """
    Args:
      encoder_feat          (tensor [bs, num_query, emded_dims])
    Return:   
      stacked_classes       (Tensor [num_layers, bs, full_num_query, num_classes])
      stacked_coords        (Tensor [num_layers, bs, full_num_query, code_size])
      segments              (list of Tensor [bs, num_anchor, H, W, code_size + num_classes + num_masks])
      proto                 (Tensor [bs, num_masks, 2*H, 2*W])
      last layer features   (Tensor [bs, embed_dims, H, W])
    """
    bs, num_feat, embed_dims = encoder_feat.shape
    assert num_feat == self.num_key, "num_feat must equal to num_levels * num_points"
    assert embed_dims == self.embed_dims, "embed_dims of encoder output must equal to that of decoder input"
    encoder_feat = encoder_feat + self.NNP_keyvalue_pos
    features,reference_points,init_reference_points,listed_features = self.decoderlayer(encoder_feat,encoder_feat)
    last_layer_features = listed_features[-1]
    segments, proto = self.segmenthead(listed_features)
    listed_classes = []
    listed_coords = []
    for layer_index in range(self.num_layers):
      if layer_index == 0:
        reference = init_reference_points
      else:
        reference = reference_points[layer_index - 1]
      reference = self.inverse_sigmoid(reference)
      output_class = self.NN_classfier(features)
      tmp = self.NN_regFiner(features)
      tmp[..., 0:2] = tmp[..., 0:2] + reference[..., 0:2]
      tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
      tmp[..., 4:5] = tmp[..., 4:5] + reference[..., 2:3]
      tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
      tmp[..., 0:1] = (tmp[..., 0:1] * (self.point_cloud_range[3] - self.point_cloud_range[0]) + self.point_cloud_range[0])
      tmp[..., 1:2] = (tmp[..., 1:2] * (self.point_cloud_range[4] - self.point_cloud_range[1]) + self.point_cloud_range[1])
      tmp[..., 4:5] = (tmp[..., 4:5] * (self.point_cloud_range[5] - self.point_cloud_range[2]) + self.point_cloud_range[2])
      output_coord = tmp
      listed_classes.append(output_class)
      listed_coords.append(output_coord)
    # stacked_classes     (Tensor [num_layers, bs, num_query, num_classes])
    # stacked_coords      (Tensor [num_layers, bs, num_query, code_size])
    # segments            (list of Tensor [bs, num_anchor, H, W, code_size + num_classes + num_masks])
    # proto               (Tensor [bs, num_masks, 2*H, 2*W])
    # last layer features (Tensor [bs, embed_dims, H, W])
    return torch.stack(listed_classes), torch.stack(listed_coords), segments, proto, last_layer_features
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

class DecoderLayer(nn.Module):
  def __init__(self,num_layers=6,\
                    full_dropout=0.1,full_num_query=40000,full_embed_dims=256,full_num_heads=8,full_num_levels=1,full_num_points=4,\
                    query_H=200,query_W=200,custom_dropout=0.1,custom_embed_dims=256,custom_num_heads=8,custom_num_levels=1,custom_num_points=4,\
                    code_size=10,device=torch.device("cpu")):
    """
    Args:
      num_layers       (int):   The number of repeatation of the layer
        Default: 6
      -----Full Attention-----
      full_dropout     (float): [Full Attention] The drop out rate
        Default: 0.1 
      full_num_query   (int):   [Full Attention] The number of query
        Default: 40000 
      full_embed_dims  (int):   [Full Attention] The embedding dimension of attention
        Default: 256
      full_num_heads   (int):   [Full Attention] The number of head
        Default: 8
      full_num_levels  (int):   [Full Attention] The number of scale levels in a single sequence
        Default: 1
      full_num_points  (int):   [Full Attention] The number of sampling points in a single level
        Default: 4
      -----Custom Attention-----
      query_H               (int):   [Custom Attention] The number of height of query grid
        Default: 200      
      query_W               (int):   [Custom Attention] The number of width of query grid
        Default: 200      
      custom_dropout        (float): [Custom Attention] The drop out rate
        Default: 0.1      
      custom_embed_dims     (int):   [Custom Attention] The embedding dimension of attention
        Default: 256
      custom_num_heads      (int):   [Custom Attention] The number of head
        Default: 8
      custom_num_levels     (int):   [Custom Attention] The number of scale levels in a single sequence
        Default: 1
      custom_num_points     (int):   [Custom Attention] The number of sampling points in a single level
        Default: 4
      -----Regression Finer-----
      code_size             (int):   [Regression Finer] The number of sampling points in a single level
        Default: 10
      -----Device-----
      device  (torch.device): The device
        Default: cpu
    """
    super().__init__()
    assert full_embed_dims == custom_embed_dims, "embed_dims for full and custom attention must be the same"
    self.num_layers         = num_layers
    self.full_dropout       = full_dropout
    self.full_num_query     = full_num_query
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
    # in original code, for each layer there is an independent regression finer. For simplification purpose, here shares one finer for all layers
    self.NN_regFiner       = RegressionFiner(embed_dims,code_size,device=device)
  def forward(self,key,value):
    """
    Args:
      key   (Tensor [bs, num_key, emded_dims]):   The key
      value (Tensor [bs, num_value, emded_dims]): The value
    Return:
      stacked_deocerlayer_feat  (Tensor [num_layers, bs, num_query, embed_dims])
      stacked_references_points (Tensor [num_layers, bs, num_query, 3])
      init_references_points    (Tensor             [bs, num_query, 3])
      listed_deocerlayer_feat   (list of Tensor [[bs, embed_dims, H, W], ... ])
    """
    bs, num_key, emded_dims = key.shape
    # self.query [1,  num_query, embed_dims]
    # ---->query [bs, num_query, embed_dims]
    query = self.query.repeat(bs,1,1)
    # self.reference_points [1,  num_query, 3]
    # ---->reference_points [bs, num_query, 3]
    reference_points = self.reference_points.repeat(bs,1,1)
    init_reference_points = reference_points
    output = query
    listed_output = []
    listed_reference_points = []
    #<---------------------------------main body
    for i in range(self.num_layers):
      reference_points_input = reference_points[..., :2].unsqueeze(2)
      fullAttn               = self.NN_fullAttn(query=output,key=output,value=output)
      addNorm1               = self.NN_addNorm1(x=fullAttn,y=output)
      custAttn               = self.NN_custAttn(query=addNorm1,key=key,value=value,reference_points=reference_points_input,spatial_shapes=self.spatial_shapes)
      addNorm2               = self.NN_addNorm2(x=custAttn,y=addNorm1)
      ffn                    = self.NN_ffn(addNorm2)
      output                 = self.NN_addNorm3(x=ffn,y=addNorm2)
      # [---update reference points
      tmp                    = self.NN_regFiner(output)
      new_reference_points = torch.zeros_like(reference_points)
      new_reference_points[..., :2]  = tmp[..., :2]  + self.inverse_sigmoid(reference_points[..., :2])
      new_reference_points[..., 2:3] = tmp[..., 4:5] + self.inverse_sigmoid(reference_points[..., 2:3]) 
      reference_points = new_reference_points.sigmoid()
      # --------------------------]
      listed_output.append(output)
      listed_reference_points.append(reference_points)
    stacked_reference_points = torch.stack(listed_reference_points)
    stacked_output = torch.stack(listed_output)
    # --------------------------
    #        [bs, num_query, embed_dims]
    # -----> [bs, embed_dims, H, W]
    for i in range(len(listed_output)):
      listed_output[i] = listed_output[i].view(bs,self.query_H,self.query_W,self.embed_dims).permute(0,3,1,2).contiguous()
    #------------------------------------------>
    # stacked output           [num_layers, bs, num_query, embed_dims]
    # stacked reference_points [num_layers, bs, num_query, 3]
    # init_reference_points                [bs, num_query, 3]
    # listed_output            [[bs, num_query, embed_dims], ... ]
    return stacked_output, stacked_reference_points, init_reference_points, listed_output
  def update_reference_points(self,reference_points,input_tensor,input_NN):
    tmp = input_NN(input_tensor)
    new_reference_points = torch.zeros_like(reference_points)
    new_reference_points[..., :2]  = tmp[..., :2]  + self.inverse_sigmoid(reference_points[..., :2])
    new_reference_points[..., 2:3] = tmp[..., 4:5] + self.inverse_sigmoid(reference_points[..., 2:3]) 
    new_reference_points = new_reference_points.sigmoid()
    return new_reference_points
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

class Classifier(nn.Module):
  def __init__(self,embed_dims=256,num_classes=10,device=torch.device("cpu")):
    super().__init__()
    self.embed_dims   = embed_dims
    self.num_classes  = num_classes
    self.device       = device
    cls_branch = []
    for _ in range(2):
      cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims, device=device))
      cls_branch.append(nn.LayerNorm(self.embed_dims,device=device))
      cls_branch.append(nn.ReLU(inplace=False).to(device))
    cls_branch.append(nn.Linear(self.embed_dims, self.num_classes,device=device))
    self.cls_branch = nn.Sequential(*cls_branch).to(device)
  def forward(self,x):
    return self.cls_branch(x)
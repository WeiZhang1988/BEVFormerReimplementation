import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, einsum

from attentions import *


class Encoder(nn.Module):
  def __init__(self,query_H,query_W,query_C,\
                    spat_num_cams=6,spat_num_zAnchors=4,spat_dropout=0.1,spat_embed_dims=256,spat_num_heads=8,spat_num_levels=4,spat_num_points=2,\
                    temp_num_sequences=2,temp_dropout=0.1,temp_embed_dims=256,temp_num_heads=8,temp_num_levels=4,temp_num_points=4,device=torch.device("cpu")):
    super().__init__()
    assert query_C == temp_embed_dims, "query_C and temp_embed_dims must be the same to simplify structure"
    self.query_H              = query_H
    self.query_W              = query_W
    self.query_C              = query_C
    self.num_query            = query_H * query_W
    self.spat_num_cams        = spat_num_cams
    self.spat_num_zAnchors    = spat_num_zAnchors
    self.spat_dropout         = spat_dropout
    self.spat_embed_dims      = spat_embed_dims
    self.spat_num_heads       = spat_num_heads
    self.spat_num_levels      = spat_num_levels
    self.spat_num_points      = spat_num_points
    self.temp_num_sequences   = temp_num_sequences
    self.temp_dropout         = temp_dropout
    self.temp_embed_dims      = temp_embed_dims
    self.temp_num_heads       = temp_num_heads
    self.temp_num_levels      = temp_num_levels
    self.temp_num_points      = temp_num_points
    self.device               = device
    embed_dims                = spat_embed_dims
    self.embed_dims           = embed_dims
    # self.NNP_query_origin [1(extends to bs), num_query, embed_dims]
    self.NNP_query_origin  = nn.Parameter(torch.ones(1,query_H*query_W,query_C,device=device)*0.98)
    # self.NNP_query_pos [1(extends to bs), num_query, embed_dims]
    self.NNP_query_pos     = nn.Parameter(torch.ones(1,query_H*query_W,temp_embed_dims,device=device)*0.95)
    # self.NNP_bev_pos [1(extends to bs), num_query, embed_dims]
    self.NNP_bev_pos       = nn.Parameter(torch.ones(1,query_H*query_W,temp_embed_dims,device=device)*0.95)
    # self.query [1(extends to bs), num_query, embed_dims]
    self.temp_query        = self.NNP_query_origin + self.NNP_query_pos
    self.NN_bevFormerLayer = BEVFormerLayer(spat_num_cams,spat_num_zAnchors,spat_dropout,spat_embed_dims,spat_num_heads,spat_num_levels,spat_num_points,\
                                            temp_num_sequences,temp_dropout,temp_embed_dims,temp_num_heads,temp_num_levels,temp_num_points,device)
    self.temp_key_hist     = [self.query for _ in range(temp_num_sequences)]
    self.temp_value_hist   = [self.query for _ in range(temp_num_sequences)]
  def forward(self,embed_features,spat_spatial_shapes,spat_reference_points):
    """
    Args:
      embed_features (Tensor [num_cams, bs, num_value(==num_key), embed_dims]): 
    """
    bev = self.NN_bevFormerLayer(embed_features,embed_features, self.temp_query,self.temp_key_hist,self.temp_value_hist,spat_spatial_shapes=spat_spatial_shapes,spat_reference_points=spat_reference_points,spat_reference_points_cam=spat_reference_points_cam,spat_bev_mask=spat_bev_mask,temp_spatial_shapes=temp_spatial_shapes,temp_reference_points=temp_reference_points)
  def cal_reference_points(self,depth,bs):
    zs = torch.linspace(0.5, depth - 0.5, self.spat_num_zAnchors, device=self.device).view(self.spat_num_zAnchors, 1, 1).expand(self.spat_num_zAnchors, self.query_H, self.query_W) / self.spat_num_zAnchors
    xs = torch.linspace(0.5, self.query_W- 0.5, self.query_W, device=self.device).view(1, 1, self.query_W).expand(self.spat_num_zAnchors, self.query_H, self.query_W) / self.query_W
    ys = torch.linspace(0.5, self.query_H - 0.5, self.query_H, device=self.device).view(1, self.query_H, 1).expand(self.spat_num_zAnchors, self.query_H, self.query_W) / self.query_H
    # ref_3d [num_zAnchors, query_H, query_W]
    # -----> [num_zAnchors, query_H, query_W, 3]
    # -----> [num_zAnchors, 3, query_H, query_W]
    # -----> [num_zAnchors, 3, query_H * query_W]
    # -----> [num_zAnchors, query_H * query_W, 3]
    # -----> [bs, num_zAnchors, query_H * query_W, 3]
    ref_3d = torch.stack((xs, ys, zs), -1)
    ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
    ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)#stop here


class BEVFormerLayer(nn.Module):
  """
  Args:
    image_shape        (list):  The shap of input image
      Default: [372,640]
    -----Spatial-----
    spat_num_cams      (int):   [spatial attention] The number of cameras
      Default: 6 
    spat_num_zAnchors  (int):   [spatial attention] The number of z anchors
      Default: 4 
    spat_dropout       (float): [spatial attention] The drop out rate
      Default: 0.1 
    spat_embed_dims    (int):   [spatial attention] The embedding dimension of attention. The same as inputs embedding dimension.
      Default: 256    
    spat_num_heads     (int):   [spatial attention] The number of heads.
      Default: 8    
    spat_num_levels    (int):   [spatial attention] The number of scale levels in a single sequence
      Default: 4    
    spat_num_points    (int):   [spatial attention] The number of sampling points in a single level
      Default: 2
    -----Temporal-----
    query_H            (int):   [temporal attention] The number of height of query grid
      Default: 200
    query_W            (int):   [temporal attention] The number of width of query grid
      Default: 200
    query_Z            (float): [temporal attention] The depth of query grid
      Default: 8.0
    query_C            (int):   [temporal attention] The number of channels of query grid
      Default: 3
    temp_num_sequences (int):   [temporal attention] The number of sequences of queries, kies and values.
      Default: 2 
    temp_dropout       (float): [temporal attention] The drop out rate
      Default: 0.1 
    temp_embed_dims    (int):   [temporal attention] The embedding dimension of attention. The same as inputs embedding dimension.
      Default: 256   
    temp_num_heads     (int):   [temporal attention] The number of heads.
      Default: 8   
    temp_num_levels    (int):   [temporal attention] The number of scale levels in a single sequence
      Default: 4   
    temp_num_points    (int):   [temporal attention] The number of sampling points in a single level
      Default: 4
    -----Device-----
    device (torch.device): The device
      Default: cpu
  """
  def __init__(self,image_shape=[372,640],
                    spat_num_cams=6,spat_num_zAnchors=4,spat_dropout=0.1,spat_embed_dims=256,spat_num_heads=8,spat_num_levels=4,spat_num_points=2,\
                    query_H=200,query_W=200,query_Z=8.0,query_C=3,temp_num_sequences=2,temp_dropout=0.1,temp_embed_dims=256,temp_num_heads=8,temp_num_levels=4,temp_num_points=4,device=torch.device("cpu")):
    super().__init__()
    assert spat_embed_dims == temp_embed_dims, "embed_dims for spatial and temperal attention must be the same"
    self.image_shape          = image_shape
    self.query_H              = query_H
    self.query_W              = query_W
    self.query_Z              = query_Z
    self.query_C              = query_C
    self.spat_num_cams        = spat_num_cams
    self.spat_num_zAnchors    = spat_num_zAnchors
    self.spat_dropout         = spat_dropout
    self.spat_embed_dims      = spat_embed_dims
    self.spat_num_heads       = spat_num_heads
    self.spat_num_levels      = spat_num_levels
    self.spat_num_points      = spat_num_points
    self.temp_num_sequences   = temp_num_sequences
    self.temp_dropout         = temp_dropout
    self.temp_embed_dims      = temp_embed_dims
    self.temp_num_heads       = temp_num_heads
    self.temp_num_levels      = temp_num_levels
    self.temp_num_points      = temp_num_points
    self.device               = device
    embed_dims                = spat_embed_dims
    self.embed_dims           = embed_dims
    # self.NNP_query_origin [1(extends to bs), num_query, embed_dims]
    self.NNP_query_origin  = nn.Parameter(torch.ones(1,query_H*query_W,query_C,device=device)*0.98)
    self.NN_projQ       = nn.Linear(query_C,temp_embed_dims).to(device)
    # self.NNP_query_pos [1(extends to bs), num_query, embed_dims]
    self.NNP_query_pos     = nn.Parameter(torch.ones(1,query_H*query_W,temp_embed_dims,device=device)*0.95)
    # self.query [1(extends to bs), num_query, embed_dims]
    self.query             = self.NN_projQ(self.NNP_query_origin) + self.NNP_query_pos
    self.NN_tempAttn    = TemporalSelfAttention(temp_num_sequences,temp_dropout,temp_embed_dims,temp_num_heads,temp_num_levels,temp_num_points,device)
    self.NN_spatAttn    = SpatialCrossAttention(spat_num_cams,spat_num_zAnchors,spat_dropout,spat_embed_dims,spat_num_heads,spat_num_levels,spat_num_points,device)
    self.NN_addNorm1    = AddAndNormLayer(None,embed_dims,device=device)
    self.NN_addNorm2    = AddAndNormLayer(None,embed_dims,device=device)
    self.NN_addNorm3    = AddAndNormLayer(None,embed_dims,device=device)
    self.NN_ffn         = nn.Linear(embed_dims,embed_dims,device=device)
    self.temp_key_hist     = [self.query for _ in range(temp_num_sequences)]
    self.temp_value_hist   = [self.query for _ in range(temp_num_sequences)]
  def forward(self,spat_key,spat_value,spat_spatial_shapes=None,spat_reference_points_cam=None,spat_bev_mask=None,temp_spatial_shapes=None):
    """
    Args:
      spat_key                    (Tensor [num_cams, bs, num_key, embed_dims]):       [spatial attention] The key
      spat_value                  (Tensor [num_cams, bs, num_value, embed_dims]):     [spatial attention] The value
      spat_spatial_shapes         (Tensor [num_levels, 2]):                           [spatial attention] The spatial shape of features in different levels
      spat_reference_points_cam   (Tensor [num_cam, bs, num_query, num_zAnchor, 2]):  [spatial attention] The image pixel ratio projected from reference points to each camera
      spat_bev_mask               (Tensor [num_cam, bs, num_query, num_zAnchor]):     [spatial attention] Which of reference_points_cam is valid
      temp_spatial_shapes         (Tensor [num_levels, 2]):                           [temporal attention] The spatial shape of features in different levels
    Returns:
      currentBEV                  (Tensor [bs, num_query, emded_dims])
    """
    _,bs,_,_ = spat_key.shape
    ref_3d, ref_2d = self.cal_reference_points(self.query_Z,bs)
    self.query = self.query.repeat(bs, 1, 1)
    for i,key in enumerate(self.temp_key_hist):
      self.temp_key_hist[i] = key.repeat(bs, 1, 1)
    for i,value in enumerate(self.temp_value_hist):
      self.temp_value_hist[i] = value.repeat(bs,1,1)
    #<------------------main body
    tempAttn   = self.NN_tempAttn(query=self.query,key_hist=self.temp_key_hist,value_hist=self.temp_value_hist,reference_points=ref_2d,spatial_shapes=temp_spatial_shapes)
    addNorm1   = self.NN_addNorm1(x=tempAttn,y=self.query)
    spatAttn   = self.NN_spatAttn(query=addNorm1,key=spat_key,value=spat_value,reference_points=ref_3d,spatial_shapes=spat_spatial_shapes,reference_points_cam=spat_reference_points_cam,bev_mask=spat_bev_mask)
    addNorm2   = self.NN_addNorm2(x=spatAttn,y=addNorm1)
    ffn        = self.NN_ffn(addNorm2)
    self.query = self.NN_addNorm3(x=ffn,y=addNorm2)
    #--------------------------->
    self.temp_key_hist.pop()
    self.temp_value_hist.pop()
    self.temp_key_hist.insert(0,self.query)
    self.temp_value_hist.insert(0,self.query)
    return self.query
  def cal_reference_points(self,depth,bs,dtype=torch.float):
    """
    Args:
      depth (float):  The depth of query grid 
      bs    (int):    The batch size
    Returns:
      ref_3d (Tensor [bs, query_H * query_W, num_zAnchors, 3]):    The 3d reference points for spatial attention
      ref_2d (Tensor [bs, query_H * query_W, temp_num_levels, 2]): The 2d reference points for temporal attention
    """
    zs = torch.linspace(0.5, depth - 0.5, self.spat_num_zAnchors, dtype=dtype, device=self.device).view(self.spat_num_zAnchors, 1, 1).expand(self.spat_num_zAnchors, self.query_H, self.query_W) / self.spat_num_zAnchors
    xs = torch.linspace(0.5, self.query_W- 0.5, self.query_W, dtype=dtype, device=self.device).view(1, 1, self.query_W).expand(self.spat_num_zAnchors, self.query_H, self.query_W) / self.query_W
    ys = torch.linspace(0.5, self.query_H - 0.5, self.query_H, dtype=dtype, device=self.device).view(1, self.query_H, 1).expand(self.spat_num_zAnchors, self.query_H, self.query_W) / self.query_H
    # ref_3d [num_zAnchors, query_H, query_W]
    # -----> [num_zAnchors, query_H, query_W, 3]
    # -----> [num_zAnchors, 3, query_H, query_W]
    # -----> [num_zAnchors, 3, query_H * query_W]
    # -----> [num_zAnchors, query_H * query_W, 3]
    # -----> [bs, num_zAnchors, query_H * query_W, 3]
    # -----> [bs, query_H * query_W, num_zAnchors, 3]
    ref_3d = torch.stack((xs, ys, zs), -1)
    ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
    ref_3d = ref_3d[None].repeat(bs, 1, 1, 1).permute(0,2,1,3).contiguous()
    ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, self.query_H - 0.5, self.query_H, dtype=dtype, device=self.device),torch.linspace(0.5, self.query_W - 0.5, self.query_W, dtype=dtype, device=self.device),indexing='ij')
    ref_y = ref_y.reshape(-1)[None] / self.query_H
    ref_x = ref_x.reshape(-1)[None] / self.query_W
    # ref_2d [1,  query_H * query_W, 2]
    # -----> [bs, query_H * query_W, 2]
    # -----> [bs, query_H * query_W, 1, 2]
    # -----> [bs, query_H * query_W, temp_num_levels, 2]
    ref_2d = torch.stack((ref_x, ref_y), -1)
    ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2).repeat(1, 1, self.temp_num_levels, 1)
    return ref_3d, ref_2d
  def sample_points(self, reference_points, point_cloud_range, lidar2img_trans):
    """
    Args:
      reference_points  (Tensor [bs, query_H * query_W, num_zAnchors, 3]):  The 3d reference points
      point_cloud_range (Tensor [6]):                                       The range of point cloud
      lidar2img_trans   (Tensor [bs, num_cams, 4, 4]):                      The lidar2image transformation matrices
    Returns:
      ref_3d (Tensor [bs, query_H * query_W, num_zAnchors, 3]): The 3d reference points for spatial attention
      ref_2d (Tensor [bs, query_H * query_W, temp_num_levels, 2]): The 2d reference points for temporal attention
    """
    num_cams = lidar2img_trans.size(1)
    assert num_cams == self.num_cams, "number of cameras must match"
    reference_points[..., 0:1] = reference_points[..., 0:1] * (point_cloud_range[3] - point_cloud_range[0]) + point_cloud_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2] * (point_cloud_range[4] - point_cloud_range[1]) + point_cloud_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3] * (point_cloud_range[5] - point_cloud_range[2]) + point_cloud_range[2]
    # reference_points [bs, query_H * query_W, num_zAnchors, 3]
    # ---------------> [bs, query_H * query_W, num_zAnchors, 4]
    # ---------------> [num_zAnchors, bs, query_H * query_W, 4]
    # ---------------> [num_zAnchors, bs, num_cams, query_H * query_W, 4, 1]
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1).permute(2, 0, 1, 3)[:, :, None, :, :].repeat(1, 1, num_cams, 1, 1).unsqueeze(-1)
    num_zAnchors, bs, _, num_query, _ = reference_points.size()[:3]
    # lidar2img_trans [bs, num_cams, 4, 4]
    # --------------> [1, bs, num_cams, 1, 4, 4]
    # --------------> [num_zAnchors, bs, num_cams, num_query, 4, 4]
    lidar2img_trans = lidar2img_trans[None, :, :, None, :, :].repeat(num_zAnchors, 1, 1, num_query, 1, 1)
    # reference_points_cam [num_zAnchors, bs, num_cams, num_query, 4]
    reference_points_cam = torch.matmul(lidar2img_trans.to(torch.float32),reference_points.to(torch.float32)).squeeze(-1)
    eps = 1e-5
    #bev_mask [num_zAnchors, bs, num_cams, num_query, 1]stop here to continue working on comment of dimensions
    bev_mask = (reference_points_cam[..., 2:3] > eps)
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
    reference_points_cam[..., 0] /= self.image_shape[1]
    reference_points_cam[..., 1] /= self.image_shape[0]
    reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
    bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0) & (reference_points_cam[..., 1:2] < 1.0) & (reference_points_cam[..., 0:1] < 1.0) & (reference_points_cam[..., 0:1] > 0.0))
    bev_mask = torch.nan_to_num(bev_mask)
    bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)
    return reference_points_cam, bev_mask
    


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
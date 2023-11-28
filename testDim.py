# to just test dimension "assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value" in MultiScaleDeformableAttention3D needs to be commented out
# in normal use, the assert should ramain

from attentions import *
from backbone import *
from encoder import *
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_backbone():
  batch_size            = 8
  num_cams              = 2
  stage_middle_channels = [64, 80, 96, 112]
  stage_out_channels    = [128, 256, 384, 512]
  num_block_per_stage   = [1, 1, 2, 2]
  num_layer_per_block   = 5


  x = torch.rand(size=(num_cams,batch_size,3,96,96)).to(device)
  backbone = BackBone(stage_middle_channels,stage_out_channels,num_block_per_stage,num_layer_per_block,device)
  res = backbone(x)
  print("bkb ", res.shape)

def test_spatial_cross_attention():
  batch_size   = 8
  num_cams     = 2
  num_zAnchors = 4
  dropout      = 0.1
  embed_dims   = 256
  num_heads    = 8
  num_levels   = 4
  num_points   = 2

  num_query    = 16
  num_key      = 16
  num_value    = 16
  query = torch.rand(size=(batch_size,num_query,embed_dims)).to(device)
  key   = torch.rand(size=(num_cams,batch_size,num_key,embed_dims)).to(device)
  value = torch.rand(size=(num_cams,batch_size,num_value,embed_dims)).to(device)

  reference_points = torch.rand(size=(batch_size,num_query,num_zAnchors,2)).to(device)
  spatial_shapes = torch.Tensor([[1,1],[2,2],[3,3],[1,2]]).to(device)
  reference_points_cam = torch.rand(num_cams, batch_size, num_query, num_zAnchors, 2).to(device)
  bev_mask = torch.rand(num_cams, batch_size, num_query, num_zAnchors).to(device)

  sca = SpatialCrossAttention(num_cams=num_cams,num_zAnchors=num_zAnchors,dropout=dropout,embed_dims=embed_dims,num_heads=num_heads,num_levels=num_levels,num_points=num_points,device=device)
  res = sca(query=query,key=key,value=value,reference_points=reference_points,spatial_shapes=spatial_shapes,reference_points_cam=reference_points_cam,bev_mask=bev_mask)
  print("sca ", res.shape)

def test_temporal_self_attention():
  batch_size     = 8
  num_sequences  = 2
  dropout        = 0.1
  embed_dims     = 256
  num_heads      = 8
  num_levels     = 4
  num_points     = 4

  num_query      = 16
  num_key        = 16
  num_value      = 16
  query = torch.rand(size=(batch_size,num_query,embed_dims)).to(device)
  key   = torch.rand(size=(batch_size,num_key,embed_dims)).to(device)
  value = torch.rand(size=(batch_size,num_value,embed_dims)).to(device)

  reference_points = torch.rand(size=(batch_size,num_query,num_levels,2)).to(device)
  spatial_shapes = torch.Tensor([[1,1],[2,2],[3,3],[1,2]]).to(device)

  tsa = TemporalSelfAttention(num_sequences=num_sequences,dropout=dropout,embed_dims=embed_dims,num_heads=num_heads,num_levels=num_levels,num_points=num_points,device=device)
  res = tsa(query,key_hist=[query,key],value_hist=[query,value],reference_points=reference_points,spatial_shapes=spatial_shapes)
  print("tsa ",res.shape)

def test_bev_former_layer():
  batch_size   = 8

  spat_num_cams       = 2
  spat_num_zAnchors   = 4
  spat_dropout        = 0.1
  spat_embed_dims     = 256
  spat_num_heads      = 8
  spat_num_levels     = 4
  spat_num_points     = 2

  query_H=20
  query_W=20
  query_Z=4
  query_C=3

  temp_num_sequences  = 2
  temp_dropout        = 0.1
  temp_embed_dims     = 256
  temp_num_heads      = 8
  temp_num_levels     = 1
  temp_num_points     = 4

  spat_num_key        = 32
  spat_num_value      = 32

  temp_num_query      = 16
  temp_num_key        = 16
  temp_num_value      = 16

  spat_key   = torch.rand(size=(spat_num_cams,batch_size,spat_num_key,spat_embed_dims)).to(device)
  spat_value = torch.rand(size=(spat_num_cams,batch_size,spat_num_value,spat_embed_dims)).to(device)

  temp_query = torch.rand(size=(batch_size,temp_num_query,temp_embed_dims)).to(device)
  temp_key   = torch.rand(size=(batch_size,temp_num_key,temp_embed_dims)).to(device)
  temp_value = torch.rand(size=(batch_size,temp_num_value,temp_embed_dims)).to(device)

  spat_reference_points = torch.rand(size=(batch_size,temp_num_query,spat_num_zAnchors,2)).to(device)
  spat_spatial_shapes = torch.Tensor([[1,2],[2,4],[3,6],[1,4]]).to(device)
  spat_lidar2img_trans = torch.rand(size=(batch_size, spat_num_cams, 4, 4)).to(device)

  temp_reference_points = torch.rand(size=(batch_size,temp_num_query,temp_num_levels,2)).to(device)
  temp_spatial_shapes = torch.Tensor([[10,10],[10,10],[10,10],[10,10]]).to(device)

  bev = BEVFormerLayer(spat_num_cams=spat_num_cams,spat_num_zAnchors=spat_num_zAnchors,spat_dropout=spat_dropout,spat_embed_dims=spat_embed_dims,spat_num_heads=spat_num_heads,spat_num_levels=spat_num_levels,spat_num_points=spat_num_points,\
                       query_H=query_H,query_W=query_W,query_Z=query_Z,query_C=query_C,temp_num_sequences=temp_num_sequences,temp_dropout=temp_dropout,temp_embed_dims=temp_embed_dims,temp_num_heads=temp_num_heads,temp_num_levels=temp_num_levels,temp_num_points=temp_num_points,device=device)
  res = bev(spat_key,spat_value,spat_spatial_shapes=spat_spatial_shapes,spat_lidar2img_trans=spat_lidar2img_trans)
  print("bev ",res.shape)


test_backbone()
test_spatial_cross_attention()
test_temporal_self_attention()
test_bev_former_layer()

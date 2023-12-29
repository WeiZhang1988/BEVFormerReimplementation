# to just test dimension "assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value" in MultiScaleDeformableAttention3D needs to be commented out
# in normal use, the assert should ramain
from tqdm import tqdm
import os

from attentions import *
from backbone import *
from encoder import *
from decoder import *
from bevformer import *
from dataset import *
from dataloader import *
from loss import *
from utils import *

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


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

def test_custom_attention():
  batch_size     = 8
  dropout        = 0.1
  embed_dims     = 256
  num_heads      = 8
  num_levels     = 1
  num_points     = 16

  num_query      = 16
  num_key        = 16
  num_value      = 16
  query = torch.rand(size=(batch_size,num_query,embed_dims)).to(device)
  key   = torch.rand(size=(batch_size,num_key,embed_dims)).to(device)
  value = torch.rand(size=(batch_size,num_value,embed_dims)).to(device)

  reference_points = torch.rand(size=(batch_size,num_query,num_levels,2)).to(device)
  spatial_shapes = torch.Tensor([[4,4]]).to(device)

  csa = CustomAttention(dropout=dropout,embed_dims=embed_dims,num_heads=num_heads,num_levels=num_levels,num_points=num_points,device=device)
  res = csa(query,key,value,reference_points=reference_points,spatial_shapes=spatial_shapes)
  print("csa ",res.shape)

def test_encoder_layer():
  batch_size   = 2

  num_layers   = 2

  spat_num_cams       = 4
  spat_num_zAnchors   = 4
  spat_dropout        = 0.1
  spat_embed_dims     = 256
  spat_num_heads      = 8
  spat_num_levels     = 2
  spat_num_points     = 2

  query_H=20
  query_W=20
  query_Z=8
  query_C=3

  temp_num_sequences  = 2
  temp_dropout        = 0.1
  temp_embed_dims     = 256
  temp_num_heads      = 8
  temp_num_levels     = 2
  temp_num_points     = 2

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
  spat_spatial_shapes = torch.Tensor([[1,16],[2,8]]).to(device)#[1,2],[2,4],[3,6],[1,4]
  spat_lidar2img_trans = torch.rand(size=(batch_size, spat_num_cams, 4, 4)).to(device)

  temp_reference_points = torch.rand(size=(batch_size,temp_num_query,temp_num_levels,2)).to(device)
  temp_spatial_shapes = torch.Tensor([[10,10],[10,10],[10,10],[10,10]]).to(device)

  enl = EncoderLayer(num_layers=num_layers,spat_num_cams=spat_num_cams,spat_num_zAnchors=spat_num_zAnchors,spat_dropout=spat_dropout,spat_embed_dims=spat_embed_dims,spat_num_heads=spat_num_heads,spat_num_levels=spat_num_levels,spat_num_points=spat_num_points,\
                       query_H=query_H,query_W=query_W,query_Z=query_Z,query_C=query_C,temp_num_sequences=temp_num_sequences,temp_dropout=temp_dropout,temp_embed_dims=temp_embed_dims,temp_num_heads=temp_num_heads,temp_num_levels=temp_num_levels,temp_num_points=temp_num_points,device=device)
  res = enl(spat_key,spat_value,spat_spatial_shapes=spat_spatial_shapes,spat_lidar2img_trans=spat_lidar2img_trans)
  print("enl ",res.shape)

def test_encoder():
  # common pars
  batch_size            = 8
  num_layers            = 2
  num_cams              = 2
  image_shape           = [96,96]
  point_cloud_range     = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
  # backbone pars
  stage_middle_channels = [64, 80, 96, 112]
  stage_out_channels    = [128, 256, 384, 512]
  num_block_per_stage   = [1, 1, 2, 2]
  num_layer_per_block   = 5
  backbone = BackBone(stage_middle_channels,stage_out_channels,num_block_per_stage,num_layer_per_block,device)
  # encoderlayer pars
  spat_num_zAnchors   = 4
  spat_dropout        = 0.1
  spat_embed_dims     = 256
  spat_num_heads      = 8
  spat_num_levels     = 2
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

  encoderlayer = EncoderLayer(num_layers=num_layers,image_shape=image_shape, point_cloud_range=point_cloud_range,\
                       spat_num_cams=num_cams,spat_num_zAnchors=spat_num_zAnchors,spat_dropout=spat_dropout,spat_embed_dims=spat_embed_dims,spat_num_heads=spat_num_heads,spat_num_levels=spat_num_levels,spat_num_points=spat_num_points,\
                       query_H=query_H,query_W=query_W,query_Z=query_Z,query_C=query_C,temp_num_sequences=temp_num_sequences,temp_dropout=temp_dropout,temp_embed_dims=temp_embed_dims,temp_num_heads=temp_num_heads,temp_num_levels=temp_num_levels,temp_num_points=temp_num_points,device=device)
  encoder = Encoder(backbone=backbone,encoderlayer=encoderlayer,device=device)


  list_leveled_images = [torch.rand(size=(num_cams, batch_size, 3, image_shape[0], image_shape[1])).to(device),
                         torch.rand(size=(num_cams, batch_size, 3, int(image_shape[0]/2),     int(image_shape[1]/2))).to(device)]
  spat_lidar2img_trans = torch.rand(size=(batch_size, num_cams, 4, 4)).to(device)
  res = encoder(list_leveled_images,spat_lidar2img_trans)
  print("enc ",res.shape)

def test_decoder_layer():
  batch_size = 8
  num_layers = 2
  full_num_query=20*20
  full_dropout=0.1
  full_embed_dims=256
  full_num_heads=8
  full_num_levels=1
  full_num_points=400
  query_H=20
  query_W=20
  custom_dropout=0.1
  custom_embed_dims=256
  custom_num_heads=8
  custom_num_levels=1
  custom_num_points=400
  code_size=10

  decoderlayer = DecoderLayer(num_layers=num_layers,full_num_query=query_H*query_W,full_dropout=full_dropout,full_embed_dims=full_embed_dims,full_num_heads=full_num_heads,full_num_levels=full_num_levels,full_num_points=full_num_points,\
                              query_H=query_H,query_W=query_W,custom_dropout=custom_dropout,custom_embed_dims=custom_embed_dims,custom_num_heads=custom_num_heads,custom_num_levels=custom_num_levels,custom_num_points=custom_num_points,\
                              code_size=code_size,device=device)

  encoder_out = torch.rand(size=(batch_size,query_H*query_W,custom_embed_dims)).to(device)
  feat,refp,init,_ = decoderlayer(encoder_out,encoder_out)
  print("del  feat ",feat.shape, "refp ",refp.shape, "init ",init.shape)

def test_decoder():
  batch_size = 8
  num_classes = 2
  num_layers = 2
  full_num_query=20*20
  full_dropout=0.1
  full_embed_dims=256
  full_num_heads=8
  full_num_levels=1
  full_num_points=400
  query_H=20
  query_W=20
  custom_dropout=0.1
  custom_embed_dims=256
  custom_num_heads=8
  custom_num_levels=1
  custom_num_points=400
  code_size=10

  anchors=[[10,13, 16,30, 33,23], [10,13, 16,30, 33,23]]
  num_masks = 32
  num_protos = 256
  channels = [custom_embed_dims, custom_embed_dims]

  decoderlayer = DecoderLayer(num_layers=num_layers,full_num_query=query_H*query_W,full_dropout=full_dropout,full_embed_dims=full_embed_dims,full_num_heads=full_num_heads,full_num_levels=full_num_levels,full_num_points=full_num_points,\
                              query_H=query_H,query_W=query_W,custom_dropout=custom_dropout,custom_embed_dims=custom_embed_dims,custom_num_heads=custom_num_heads,custom_num_levels=custom_num_levels,custom_num_points=custom_num_points,\
                              code_size=code_size,device=device)
  segmenthead = Segment(nc=num_classes, cs=code_size, anchors=anchors, nm=num_masks, npr=num_protos, ch=channels)
  decoder = Decoder(num_classes=num_classes,decoderlayer=decoderlayer,segmenthead=segmenthead,device=device)

  encoder_out = torch.rand(size=(batch_size,query_H*query_W,custom_embed_dims)).to(device)

  cls, crd, segments, proto = decoder(encoder_out)
  print("dec  cls segment[0] segment[1] proto",cls.shape,"crd ",crd.shape,segments[0].shape,segments[1].shape,proto.shape)

def test_bevformer():
  # common pars
  batch_size            = 8
  num_layers            = 2
  num_cams              = 2
  image_shape           = [96,96]
  point_cloud_range     = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
  # backbone pars
  stage_middle_channels = [64, 80, 96, 112]
  stage_out_channels    = [128, 256, 384, 512]
  num_block_per_stage   = [1, 1, 2, 2]
  num_layer_per_block   = 5
  backbone = BackBone(stage_middle_channels,stage_out_channels,num_block_per_stage,num_layer_per_block,device)
  #---------------------------------------------------------------------------------------------------------
  # encoderlayer pars
  spat_num_zAnchors   = 4
  spat_dropout        = 0.1
  spat_embed_dims     = 256
  spat_num_heads      = 8
  spat_num_levels     = 2
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
  encoderlayer = EncoderLayer(num_layers=num_layers,image_shape=image_shape, point_cloud_range=point_cloud_range,\
                       spat_num_cams=num_cams,spat_num_zAnchors=spat_num_zAnchors,spat_dropout=spat_dropout,spat_embed_dims=spat_embed_dims,spat_num_heads=spat_num_heads,spat_num_levels=spat_num_levels,spat_num_points=spat_num_points,\
                       query_H=query_H,query_W=query_W,query_Z=query_Z,query_C=query_C,temp_num_sequences=temp_num_sequences,temp_dropout=temp_dropout,temp_embed_dims=temp_embed_dims,temp_num_heads=temp_num_heads,temp_num_levels=temp_num_levels,temp_num_points=temp_num_points,device=device)
  encoder = Encoder(backbone=backbone,encoderlayer=encoderlayer,device=device)
  #--------------------------------------------------------------------------------------------------------
  num_classes = 2
  num_layers = 2
  full_dropout=0.1
  full_embed_dims=256
  full_num_heads=8
  full_num_levels=1
  full_num_points=400
  custom_dropout=0.1
  custom_embed_dims=256
  custom_num_heads=8
  custom_num_levels=1
  custom_num_points=400
  code_size=10
  anchors=[[10,13, 16,30, 33,23], [10,13, 16,30, 33,23]]
  num_masks = 32
  num_protos = 256
  channels = [custom_embed_dims, custom_embed_dims]
  decoderlayer = DecoderLayer(num_layers=num_layers,full_num_query=query_H*query_W,full_dropout=full_dropout,full_embed_dims=full_embed_dims,full_num_heads=full_num_heads,full_num_levels=full_num_levels,full_num_points=full_num_points,\
                              query_H=query_H,query_W=query_W,custom_dropout=custom_dropout,custom_embed_dims=custom_embed_dims,custom_num_heads=custom_num_heads,custom_num_levels=custom_num_levels,custom_num_points=custom_num_points,\
                              code_size=code_size,device=device)
  segmenthead = Segment(nc=num_classes, cs=code_size, anchors=anchors, nm=num_masks, npr=num_protos, ch=channels)
  decoder = Decoder(num_classes=num_classes,decoderlayer=decoderlayer,segmenthead=segmenthead,device=device)
  #----------------------------------------------------------------------------------------------------
  bevformer = BEVFormer(encoder=encoder,decoder=decoder,lr=1e-4,device=device)
  #-----------------------------------------------------------------------------------------------------
  list_leveled_images = [torch.rand(size=(num_cams, batch_size, 3, image_shape[0], image_shape[1])).to(device),
                         torch.rand(size=(num_cams, batch_size, 3, int(image_shape[0]/2),     int(image_shape[1]/2))).to(device)]
  spat_lidar2img_trans = torch.rand(size=(batch_size, num_cams, 4, 4)).to(device)
  inputs = {'list_leveled_images': list_leveled_images,'spat_lidar2img_trans': spat_lidar2img_trans}
  cls, crd, segments, proto = bevformer(inputs)
  print("bev  cls segment[0] segment[1] proto",cls.shape,"crd ",crd.shape,segments[0].shape,segments[1].shape,proto.shape)

def test_loss():
  # common pars
  batch_size            = 2
  num_layers            = 2
  num_cams              = 4
  image_shape           = [96,96]
  point_cloud_range     = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
  num_gpu     = torch.cuda.device_count()  # number of CUDA devices
  num_threads = min([os.cpu_count() // max(num_gpu, 1), batch_size if batch_size > 1 else 0, 8])
  # backbone pars
  stage_middle_channels = [64, 80, 96, 112]
  stage_out_channels    = [128, 256, 384, 512]
  num_block_per_stage   = [1, 1, 2, 2]
  num_layer_per_block   = 5
  backbone = BackBone(stage_middle_channels,stage_out_channels,num_block_per_stage,num_layer_per_block,device)
  #------------------------------------------------------------------------------------------------------------
  # encoderlayer pars
  spat_num_zAnchors   = 4
  spat_dropout        = 0.1
  spat_embed_dims     = 256
  spat_num_heads      = 8
  spat_num_levels     = 2
  spat_num_points     = 2
  query_H=100
  query_W=100
  query_Z=8
  query_C=3
  temp_num_sequences  = 2
  temp_dropout        = 0.1
  temp_embed_dims     = 256
  temp_num_heads      = 8
  temp_num_levels     = 2
  temp_num_points     = 4
  encoderlayer = EncoderLayer(num_layers=num_layers,image_shape=image_shape, point_cloud_range=point_cloud_range,\
                       spat_num_cams=num_cams,spat_num_zAnchors=spat_num_zAnchors,spat_dropout=spat_dropout,spat_embed_dims=spat_embed_dims,spat_num_heads=spat_num_heads,spat_num_levels=spat_num_levels,spat_num_points=spat_num_points,\
                       query_H=query_H,query_W=query_W,query_Z=query_Z,query_C=query_C,temp_num_sequences=temp_num_sequences,temp_dropout=temp_dropout,temp_embed_dims=temp_embed_dims,temp_num_heads=temp_num_heads,temp_num_levels=temp_num_levels,temp_num_points=temp_num_points,device=device)
  encoder = Encoder(backbone=backbone,encoderlayer=encoderlayer,device=device)
  #------------------------------------------------------------------------------------------------------------
  num_classes = 23
  num_layers = 2
  full_dropout=0.1
  full_embed_dims=256
  full_num_heads=8
  full_num_levels=1
  full_num_points=400
  custom_dropout=0.1
  custom_embed_dims=256
  custom_num_heads=8
  custom_num_levels=1
  custom_num_points=400
  code_size=5
  anchors=[[10,13, 16,30, 33,23], [10,13, 16,30, 33,23]]
  num_masks = 32
  num_protos = 256
  channels = [custom_embed_dims, custom_embed_dims]
  decoderlayer = DecoderLayer(num_layers=num_layers,full_num_query=query_H*query_W,full_dropout=full_dropout,full_embed_dims=full_embed_dims,full_num_heads=full_num_heads,full_num_levels=full_num_levels,full_num_points=full_num_points,\
                              query_H=query_H,query_W=query_W,custom_dropout=custom_dropout,custom_embed_dims=custom_embed_dims,custom_num_heads=custom_num_heads,custom_num_levels=custom_num_levels,custom_num_points=custom_num_points,\
                              code_size=code_size,device=device)
  segmenthead = Segment(nc=num_classes, cs=code_size, anchors=anchors, nm=num_masks, npr=num_protos, ch=channels)
  decoder = Decoder(num_classes=num_classes,decoderlayer=decoderlayer,segmenthead=segmenthead,device=device)
  #------------------------------------------------------------------------------------------------------------
  bevformer = BEVFormer(encoder=encoder,decoder=decoder,lr=1e-4,device=device)
  #------------------------------------------------------------------------------------------------------------
  anchor_t = 4
  eps = 1e-5
  weight_box_loss = 1.0
  weight_obj_loss = 1.0
  weight_cls_loss = 1.0
  generator = torch.Generator()
  generator.manual_seed(6148914691236517205 - 1)
  dataset = BEVDataset(img_dir='./data/images', label_dir='./data/labels', cache_dir='./data/cache', lidar2img_trans=torch.tile(torch.eye(4),(3,1,1)), num_levels=2, batch_size=batch_size, num_threads=num_threads, bev_size=(640,640), overlap=False)
  loader = InfiniteDataLoader(dataset=dataset,batch_size=batch_size,num_workers=num_threads,pin_memory=True,collate_fn=BEVDataset.collate_fn,shuffle=True,drop_last=False,)
  bevloss = BEVLoss(anchors=anchors,anchor_t=anchor_t,num_classes=num_classes,num_masks=num_masks,eps=eps,weight_box_loss=weight_box_loss,weight_obj_loss=weight_obj_loss,weight_cls_loss=weight_cls_loss, device=device)
  #------------------------------------------------------------------------------------------------------------
  list_leveled_images = [torch.rand(size=(num_cams, batch_size, 3, image_shape[0], image_shape[1])).to(device),
                         torch.rand(size=(num_cams, batch_size, 3, int(image_shape[0]/2),     int(image_shape[1]/2))).to(device)]
  spat_lidar2img_trans = torch.rand(size=(batch_size, num_cams, 4, 4)).to(device)
  inputs = {'list_leveled_images': list_leveled_images,'spat_lidar2img_trans': spat_lidar2img_trans}
  cls, crd, segments, proto = bevformer(inputs)

  loop = tqdm(loader, leave=True)
  for batch_idx, (imgs_outs, lidar2img_transes, labels_outs, masks_outs) in enumerate(loop):
    loss, loss_items = bevloss(segments, proto, labels_outs.to(device), masks=masks_outs.to(device).float())
    print("loss ",loss)
    print("loss_items ",loss_items)



def test_cache_labels():
  bev_data = BEVDataset(img_dir='./data/images', label_dir='./data/labels', cache_dir='./data/cache', lidar2img_trans=torch.tile(torch.eye(4),(3,1,1)), num_levels=2, batch_size=16, num_threads=1, bev_size=(640,640), overlap=False)
  print(bev_data.cache_labels().items())

def test_load_image():
  bev_data = BEVDataset(img_dir='./data/images', label_dir='./data/labels', cache_dir='./data/cache', lidar2img_trans=torch.tile(torch.eye(4),(3,1,1)), num_levels=2, batch_size=16, num_threads=1, bev_size=(640,640), overlap=False)
  image, trans, h, w = bev_data.load_image(0)
  print("image ",len(image))
  print("trans", trans.shape)
  print("image, trans, h, w \n", (image, trans, h, w))

def test_get_item():
  bev_data = BEVDataset(img_dir='./data/images', label_dir='./data/labels', cache_dir='./data/cache', lidar2img_trans=torch.tile(torch.eye(4),(3,1,1)), num_levels=2, batch_size=16, num_threads=1, bev_size=(640,640), overlap=False)
  imgs_out, lidar2img_trans, labels_out, masks_out = bev_data.__getitem__(0)
  for im in imgs_out:
    print("im ",im.shape)
  print("lidar2img_trans", lidar2img_trans.shape)
  print("labels_out", labels_out.shape)
  print("masks_out", masks_out.shape)

def test_dataloader():
  dataset = BEVDataset(img_dir='./data/images', label_dir='./data/labels', cache_dir='./data/cache', lidar2img_trans=torch.tile(torch.eye(4),(3,1,1)), num_levels=2, batch_size=16, num_threads=1, bev_size=(640,640), overlap=False)
  loader = InfiniteDataLoader(dataset=dataset,batch_size=2,num_workers=2,pin_memory=True,collate_fn=BEVDataset.collate_fn,shuffle=True,drop_last=False,)
  loop = tqdm(loader, leave=True)
  for batch_idx, (imgs_outs, lidar2img_transes, labels_outs, masks_outs) in enumerate(loop):
    print("batch id -------------------", batch_idx)
    print("imgs_outs ",imgs_outs.shape)
    print("lidar2img_transes ", lidar2img_transes.shape)
    print("labels_outs ", labels_outs.shape)
    print("masks_outs ", masks_outs.shape)

def test_dims():
  test_backbone()
  test_spatial_cross_attention()
  test_temporal_self_attention()
  test_custom_attention()
  test_encoder_layer()
  test_encoder()
  test_decoder_layer()
  test_decoder()
  test_bevformer()

def test_dataset():
  test_cache_labels()
  test_load_image()
  test_get_item()

if __name__ == "__main__":
  test_encoder_layer()


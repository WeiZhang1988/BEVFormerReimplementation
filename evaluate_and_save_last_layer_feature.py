import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from utils import load_checkpoint
from backbone import BackBone
from encoder import EncoderLayer, Encoder
from heads import Segment
from decoder import DecoderLayer, Decoder
from bevformer import BEVFormer
from dataset import BEVDataset
from dataloader import InfiniteDataLoader

#<<<<<<<<<<<< system config
torch.manual_seed(123)
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 2e-5
weight_decay  = 0
load_model    = True
checkpoint    = './model/model.pth.tar'
file_path     = './data/last_layer_feature/last_layer_feature.np'
#system config >>>>>>>>>>>>

#<<<<<<<<<<<< model config
from attentions import *
from backbone import *
from encoder import *
from decoder import *
from bevformer import *
from dataset import *
from dataloader import *
from loss import *
from utils import *
import torch
#<-- common parameters
common_embed_dims            = 256
common_point_cloud_range     = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
common_num_inputs_levels     = 1 
common_query_H               = 20
common_query_W               = 20
common_num_classes           = 23
common_code_size             = 5
common_anchors               = [[10,13, 16,30, 33,23], [10,13, 16,30, 33,23]]
common_num_anchor            = len(common_anchors)
common_num_masks             = 32
#common parameters -->
#<-- backbone parameters
backbone_stage_middle_channels = [64, 80, 96, 112]
backbone_stage_out_channels    = [128, 256, 384, 512]
backbone_num_block_per_stage   = [1, 1, 2, 2]
backbone_num_layer_per_block   = 5
#backbone parameters -->
#<-- encoder layer parameters
encoder_num_layer           = 1
encoder_cam_image_shape     = [256,256] 
encoder_point_cloud_range   = common_point_cloud_range
encoder_num_cams            = 4 
encoder_spat_num_zAnchors   = 4
encoder_spat_dropout        = 0.1
encoder_spat_embed_dims     = common_embed_dims
encoder_spat_num_heads      = 8
encoder_spat_num_levels     = common_num_inputs_levels
encoder_spat_num_points     = 2
encoder_query_H             = common_query_H
encoder_query_W             = common_query_W
encoder_query_Z             = 8 # confusing name
encoder_query_C             = 3
encoder_temp_num_sequences  = 2
encoder_temp_dropout        = 0.1
encoder_temp_embed_dims     = common_embed_dims
encoder_temp_num_heads      = 8
encoder_temp_num_levels     = common_num_inputs_levels
encoder_temp_num_points     = 2
#encoder layer parameters -->
#<-- decoder layer parameters
decoder_num_classes         = common_num_classes
decoder_point_cloud_range   = common_point_cloud_range
decoder_num_layer           = common_num_anchor
decoder_full_dropout        = 0.1
decoder_full_num_query      = common_query_H * common_query_W
decoder_full_embed_dims     = common_embed_dims
decoder_full_num_heads      = 8
decoder_full_num_levels     = common_num_inputs_levels
decoder_full_num_points     = decoder_full_num_query
decoder_query_H             = common_query_H
decoder_query_W             = common_query_W
decoder_custom_dropout      = 0.1
decoder_custom_embed_dims   = common_embed_dims
decoder_custom_num_heads    = 8
decoder_custom_num_levels   = common_num_inputs_levels
decoder_custom_num_points   = common_query_H * common_query_W
decoder_code_size           = common_code_size
#decoder layer parameters -->
#<-- segment head parameters
seg_num_classes = common_num_classes
seg_code_size   = common_code_size
seg_anchors     = common_anchors
seg_num_masks   = common_num_masks
seg_num_protos  = 256
seg_channels    = [decoder_custom_embed_dims, decoder_custom_embed_dims] # Note: the lenght of seg_channels must match the common_num_anchor
#segment head parameters -->
#<-- dataset and dataloader parameters
data_img_dir              = "./data/images"
data_label_dir            = "./data/labels"
data_cache_dir            = "./data/cache"
data_lidar2image_trans    = torch.stack([torch.tensor([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,2.4],[0.,0.,0.,1.]]),
                                         torch.tensor([[0.,-1.,0.,0.],[1.,0.,0.,0.],[0.,0.,1.,2.4],[0.,0.,0.,1.]]),
                                         torch.tensor([[-1.,0.,0.,0.],[0.,-1.,0.,0.],[0.,0.,1.,2.4],[0.,0.,0.,1.]]),
                                         torch.tensor([[0.,1.,0.,0.],[-1.,0.,0.,0.],[0.,0.,1.,2.4],[0.,0.,0.,1.]])],dim=0)
data_num_levels           = common_num_inputs_levels
data_batch_size           = 1
data_num_gpu              = torch.cuda.device_count()  # number of CUDA devices
data_num_threads          = 1
data_bev_size             = (256,256)
data_overlap              = False
data_pin_memory           = True
data_shuffle              = True
data_drop_last            = False
#dataset and dataloader parameters -->
#<-- loss parameters
loss_anchors         = common_anchors
loss_anchor_t        = 4
loss_num_classes     = common_num_classes
loss_num_masks       = common_num_masks
loss_eps             = 1e-5
loss_weight_box_loss = 1.0
loss_weight_obj_loss = 1.0
loss_weight_cls_loss = 1.0
#loss parameters -->
#model config >>>>>>>>>>>>>

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic=True
torch.cuda.synchronize()

def evaluate_and_save_last_layer_feature(model, lidar2img_transes=data_lidar2image_trans, image_path='./data/images', output_path='./data/last_layer_feature'):
  model.eval()
  lidar2img_transes = torch.unsqueeze(lidar2img_transes,dim=0).to(device)
  image_list = glob.glob(image_path+'/*')
  assert image_list, f'No images found'
  image_dic = {}
  for image_file in image_list:
    image = torch.tensor(np.expand_dims(np.transpose(np.array(Image.open(image_file)),(2,0,1)),axis=0),dtype=torch.float).to(device)
    image_body = image_file.split('.')[-2]
    image_frame_ID = image_body.split('_')[-2].split('/')[-1]
    if image_frame_ID in image_dic:
      image_dic[image_frame_ID].append(image)
    else:
      image_dic[image_frame_ID] = [image]
  image_ordered = collections.OrderedDict(sorted(image_dic.items()))
  for key, value in image_ordered.items():
    output_name = output_path + '/' + key
    images = torch.stack(value,dim=0)
    model_inputs = {'list_leveled_images': [images],'spat_lidar2img_trans': lidar2img_transes}
    _, _, _, _, last_layer_features = model(model_inputs)
    np.save(output_name, last_layer_features.cpu().detach().numpy())
  model.train()

def main():
  backbone = BackBone(backbone_stage_middle_channels,
                      backbone_stage_out_channels,
                      backbone_num_block_per_stage,
                      backbone_num_layer_per_block,
                      device)
  encoderlayer = EncoderLayer(num_layers=encoder_num_layer,
                              image_shape=encoder_cam_image_shape, 
                              point_cloud_range=encoder_point_cloud_range,
                              spat_num_cams=encoder_num_cams,
                              spat_num_zAnchors=encoder_spat_num_zAnchors,
                              spat_dropout=encoder_spat_dropout,
                              spat_embed_dims=encoder_spat_embed_dims,
                              spat_num_heads=encoder_spat_num_heads,
                              spat_num_levels=encoder_spat_num_levels,
                              spat_num_points=encoder_spat_num_points,
                              query_H=encoder_query_H,
                              query_W=encoder_query_W,
                              query_Z=encoder_query_Z,
                              query_C=encoder_query_C,
                              temp_num_sequences=encoder_temp_num_sequences,
                              temp_dropout=encoder_temp_dropout,
                              temp_embed_dims=encoder_temp_embed_dims,
                              temp_num_heads=encoder_temp_num_heads,
                              temp_num_levels=encoder_temp_num_levels,
                              temp_num_points=encoder_temp_num_points,
                              device=device)
  encoder = Encoder(backbone=backbone,
                    encoderlayer=encoderlayer,
                    device=device)
  decoderlayer = DecoderLayer(num_layers=decoder_num_layer,
                              full_dropout=decoder_full_dropout,
                              full_num_query=decoder_full_num_query,
                              full_embed_dims=decoder_full_embed_dims,
                              full_num_heads=decoder_full_num_heads,
                              full_num_levels=decoder_full_num_levels,
                              full_num_points=decoder_full_num_points,
                              query_H=decoder_query_H,
                              query_W=decoder_query_W,
                              custom_dropout=decoder_custom_dropout,
                              custom_embed_dims=decoder_custom_embed_dims,
                              custom_num_heads=decoder_custom_num_heads,
                              custom_num_levels=decoder_custom_num_levels,
                              custom_num_points=decoder_custom_num_points,
                              code_size=decoder_code_size,
                              device=device)
  segmenthead = Segment(nc=seg_num_classes, 
                        cs=seg_code_size, 
                        anchors=seg_anchors, 
                        nm=seg_num_masks, 
                        npr=seg_num_protos, 
                        ch=seg_channels, 
                        device=device)
  decoder = Decoder(num_classes=decoder_num_classes,
                    decoderlayer=decoderlayer,
                    segmenthead=segmenthead,
                    device=device)
  bevformer = BEVFormer(encoder=encoder,
                        decoder=decoder,
                        lr=learning_rate,
                        device=device)
  del backbone, encoderlayer, encoder, decoderlayer, segmenthead, decoder
  optimizer = optim.Adam(bevformer.parameters(),
                         lr=learning_rate,
                         weight_decay=weight_decay)
  eval_dataset = BEVDataset(img_dir=data_img_dir, 
                             label_dir=data_label_dir, 
                             cache_dir=data_cache_dir, 
                             lidar2img_trans=data_lidar2image_trans, 
                             num_levels=data_num_levels,
                             batch_size=data_batch_size, 
                             num_threads=data_num_threads, 
                             bev_size=data_bev_size, 
                             overlap=data_overlap)
  eval_dataloader = InfiniteDataLoader(dataset=eval_dataset,
                                        batch_size=data_batch_size,
                                        num_workers=data_num_threads,
                                        pin_memory=data_pin_memory,
                                        collate_fn=BEVDataset.collate_fn,
                                        shuffle=False,
                                        drop_last=False,)

  
  if os.path.exists(checkpoint) and load_checkpoint:
    load_checkpoint(torch.load(checkpoint),bevformer,optimizer)
  
  evaluate_and_save_last_layer_feature(bevformer)

if __name__ == "__main__":
  main()
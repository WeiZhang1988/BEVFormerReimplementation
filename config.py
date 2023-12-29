#<<<<<<<<<<<< system config
import os
import time
import torch
import torch.optim as optim
torch.manual_seed(123)
torch.autograd.set_detect_anomaly(True)
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device        = torch.device("cpu")
learning_rate = 2e-5
weight_decay  = 0
num_epochs    = 300
save_freq     = 50 #save every 10 epochs
load_model    = False
checkpoint    = './model/model.pth.tar'
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
data_lidar2image_trans    = torch.tile(torch.eye(4),(encoder_num_cams,1,1))
data_num_levels           = common_num_inputs_levels
data_batch_size           = 2
data_num_gpu              = torch.cuda.device_count()  # number of CUDA devices
data_num_threads          = min([os.cpu_count() // max(data_num_gpu, 1), data_batch_size if data_batch_size > 1 else 0, 8])
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
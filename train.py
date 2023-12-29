import os
import time
import torch
import torch.optim as optim

from utils import load_checkpoint, save_checkpoint
from backbone import BackBone
from encoder import EncoderLayer, Encoder
from heads import Segment
from decoder import DecoderLayer, Decoder
from bevformer import BEVFormer
from dataset import BEVDataset
from dataloader import InfiniteDataLoader
from loss import BEVLoss
from tqdm import tqdm

import config

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
torch.autograd.set_detect_anomaly(True)

def train_fn(train_loader, model, optimizer, loss_fn):
  loop = tqdm(train_loader, leave=True)
  mean_loss = []
  for batch_idx, (imgs_outs, lidar2img_transes, labels_outs, masks_outs) in enumerate(loop):
    imgs_outs, lidar2img_transes, labels_outs, masks_outs = imgs_outs.to(config.device), lidar2img_transes.to(config.device), labels_outs.to(config.device), masks_outs.to(config.device) 
    imgs_outs = imgs_outs.permute([1,0,2,3,4]).contiguous()
    model_inputs = {'list_leveled_images': [imgs_outs],'spat_lidar2img_trans': lidar2img_transes}
    cls, crd, segments, proto = model(model_inputs)
    loss, loss_items = loss_fn(segments, proto, labels_outs, masks=masks_outs)
    mean_loss.append(loss.item())
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    loop.set_postfix(loss=loss.item())
  avg_loss = sum(mean_loss)/len(mean_loss)
  print(f"trainning mean loss was {avg_loss}")
  return avg_loss

def eval_fn(eval_loader, model, loss_fn):
  model.eval()
  loop = tqdm(eval_loader, leave=True)
  mean_loss = []
  for batch_idx, (imgs_outs, lidar2img_transes, labels_outs, masks_outs) in enumerate(loop):
    imgs_outs.to(config.device), lidar2img_transes.to(config.device), labels_outs.to(config.device), masks_outs.to(config.device)
    cls, crd, segments, proto = model({'list_leveled_images': [imgs_outs],'spat_lidar2img_trans': lidar2img_transes})
    loss, loss_items = loss_fn(segments, proto, labels_outs, masks=masks_outs.float())
    mean_loss.append(loss.item())
    loop.set_postfix(loss=loss.item())
  print(f"trainning mean loss was {sum(mean_loss)/len(mean_loss)}")
  model.train()

def main():
  backbone = BackBone(config.backbone_stage_middle_channels,
                      config.backbone_stage_out_channels,
                      config.backbone_num_block_per_stage,
                      config.backbone_num_layer_per_block,
                      config.device)
  encoderlayer = EncoderLayer(num_layers=config.encoder_num_layer,
                              image_shape=config.encoder_cam_image_shape, 
                              point_cloud_range=config.encoder_point_cloud_range,
                              spat_num_cams=config.encoder_num_cams,
                              spat_num_zAnchors=config.encoder_spat_num_zAnchors,
                              spat_dropout=config.encoder_spat_dropout,
                              spat_embed_dims=config.encoder_spat_embed_dims,
                              spat_num_heads=config.encoder_spat_num_heads,
                              spat_num_levels=config.encoder_spat_num_levels,
                              spat_num_points=config.encoder_spat_num_points,
                              query_H=config.encoder_query_H,
                              query_W=config.encoder_query_W,
                              query_Z=config.encoder_query_Z,
                              query_C=config.encoder_query_C,
                              temp_num_sequences=config.encoder_temp_num_sequences,
                              temp_dropout=config.encoder_temp_dropout,
                              temp_embed_dims=config.encoder_temp_embed_dims,
                              temp_num_heads=config.encoder_temp_num_heads,
                              temp_num_levels=config.encoder_temp_num_levels,
                              temp_num_points=config.encoder_temp_num_points,
                              device=config.device)
  encoder = Encoder(backbone=backbone,
                    encoderlayer=encoderlayer,
                    device=config.device)
  decoderlayer = DecoderLayer(num_layers=config.decoder_num_layer,
                              full_dropout=config.decoder_full_dropout,
                              full_num_query=config.decoder_full_num_query,
                              full_embed_dims=config.decoder_full_embed_dims,
                              full_num_heads=config.decoder_full_num_heads,
                              full_num_levels=config.decoder_full_num_levels,
                              full_num_points=config.decoder_full_num_points,
                              query_H=config.decoder_query_H,
                              query_W=config.decoder_query_W,
                              custom_dropout=config.decoder_custom_dropout,
                              custom_embed_dims=config.decoder_custom_embed_dims,
                              custom_num_heads=config.decoder_custom_num_heads,
                              custom_num_levels=config.decoder_custom_num_levels,
                              custom_num_points=config.decoder_custom_num_points,
                              code_size=config.decoder_code_size,
                              device=config.device)
  segmenthead = Segment(nc=config.seg_num_classes, 
                        cs=config.seg_code_size, 
                        anchors=config.seg_anchors, 
                        nm=config.seg_num_masks, 
                        npr=config.seg_num_protos, 
                        ch=config.seg_channels, 
                        device=config.device)
  decoder = Decoder(num_classes=config.decoder_num_classes,
                    decoderlayer=decoderlayer,
                    segmenthead=segmenthead,
                    device=config.device)
  bevformer = BEVFormer(encoder=encoder,
                        decoder=decoder,
                        lr=config.learning_rate,
                        device=config.device)
  bevloss = BEVLoss(anchors=config.loss_anchors,
                    anchor_t=config.loss_anchor_t,
                    num_classes=config.loss_num_classes,
                    num_masks=config.loss_num_masks,
                    eps=config.loss_eps,
                    weight_box_loss=config.loss_weight_box_loss,
                    weight_obj_loss=config.loss_weight_obj_loss,
                    weight_cls_loss=config.loss_weight_cls_loss, 
                    device=config.device)
  optimizer = optim.Adam(bevformer.parameters(),
                         lr=config.learning_rate,
                         weight_decay=config.weight_decay)
  train_dataset = BEVDataset(img_dir=config.data_img_dir, 
                             label_dir=config.data_label_dir, 
                             cache_dir=config.data_cache_dir, 
                             lidar2img_trans=config.data_lidar2image_trans, 
                             num_levels=config.data_num_levels,
                             batch_size=config.data_batch_size, 
                             num_threads=config.data_num_threads, 
                             bev_size=config.data_bev_size, 
                             overlap=config.data_overlap)
  train_dataloader = InfiniteDataLoader(dataset=train_dataset,
                                        batch_size=config.data_batch_size,
                                        num_workers=config.data_num_threads,
                                        pin_memory=config.data_pin_memory,
                                        collate_fn=BEVDataset.collate_fn,
                                        shuffle=config.data_shuffle,
                                        drop_last=config.data_drop_last,)
 
  # if os.path.exists(config.checkpoint) and config.load_checkpoint:
  #   load_checkpoint(torch.load(config.checkpoint),bevformer,optimizer)
  
  for epoch in range(config.num_epochs):
    print(f"train the {epoch}th epoch")
    avg_loss = train_fn(train_dataloader, bevformer, optimizer, bevloss)
    # if avg_loss > 0.9 or epoch % config.save_freq == 0:
    #   checkpoint = {"state_dict": bevformer.state_dict(),"optimizer": optimizer.state_dict(),}
    #   save_checkpoint(checkpoint, checkpoint=config.checkpoint)
    #   time.sleep(3)

if __name__ == "__main__":
  main()
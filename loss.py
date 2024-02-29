import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def smooth_BCE(eps=0.1): 
  return 1.0 - 0.5 * eps, 0.5 * eps

def xywh2xyxy(x):
  # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
  y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
  y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
  y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
  y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
  y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
  return y

def crop_mask(masks, boxes):
  """
  "Crop" predicted masks by zeroing out everything not in the predicted bbox.
  Args:
      - masks should be a size [n, h, w] tensor of masks
      - boxes should be a size [n, 4] tensor of bbox coords in relative point form
  """
  n, h, w = masks.shape
  x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
  r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
  c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)
  return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
  # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)
  # Get the coordinates of bounding boxes
  if xywh:  # transform from xywh to xyxy
    (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
    w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
  else:  # x1, y1, x2, y2 = box1
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
    w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)
  # Intersection area
  inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)
  # Union Area
  union = w1 * h1 + w2 * h2 - inter + eps
  # IoU
  iou = inter / union
  if CIoU or DIoU or GIoU:
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
    if CIoU or DIoU:  # Distance or Complete IoU 
      c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
      rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
      if CIoU: 
        v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
        with torch.no_grad():
          alpha = v / (v - iou + (1 + eps))
        return iou - (rho2 / c2 + v * alpha)  # CIoU
      return iou - rho2 / c2  # DIoU
    c_area = cw * ch + eps  # convex area
    return iou - (c_area - union) / c_area  # GIoU 
  return iou  # IoU

class BEVLoss(nn.Module):
  count = 0
  def __init__(self,anchors=[[10,13, 16,30, 33,23], [10,13, 16,30, 33,23]],anchor_t=4,num_classes=23,num_masks=32,eps=1e-5,weight_box_loss=1.0,weight_obj_loss=1.0,weight_cls_loss=1.0, device=torch.device("cpu")):
    super().__init__()
    self.anchor_t    = anchor_t
    self.num_anchors = len(anchors[0]) // 2    # number of anchors
    self.num_classes = num_classes             # number of classes
    self.num_layers  = len(anchors)            # number of layers
    self.anchors     = torch.tensor(anchors).to(device).float().view(self.num_layers, -1, 2)
    self.num_masks   = num_masks             # number of masks
    self.weight_box_loss = weight_box_loss
    self.weight_obj_loss = weight_obj_loss
    self.weight_cls_loss = weight_cls_loss
    self.device = device
    self.overlap = True
    self.sort_obj_iou = False
    # Define criteria
    self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
    self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
    # Class label smoothing
    self.cp, self.cn = smooth_BCE(eps=eps)  # positive, negative BCE targets
    self.balance = {2: [4.0, 4.0]}.get(self.num_layers, [4.0, 4.0, 4.0, 4.0, 4.0])
    self.gr = 1.0
    BEVLoss.count += 1
  def __del__(self):
    BEVLoss.count -= 1
  def forward(self, segments, proto, targets, masks):
    bs, nm, mask_h, mask_w = proto.shape # batch size, number of masks, mask height, mask width
    lcls = torch.zeros(1, device=self.device)
    lbox = torch.zeros(1, device=self.device)
    lobj = torch.zeros(1, device=self.device)
    lseg = torch.zeros(1, device=self.device)
    print("targets value ", targets)
    tcls, tbox, indices, anchors, tidxs, xywhn = self.build_targets(segments, targets)  # targets
    print("segments shape ", len(segments), "---", segments[0].shape)
    print("targets shape ", targets.shape)
    print("tcls shape ", len(tcls), "---", tcls[0].shape)
    print("tbox shape ", len(tbox), "---", tbox[0].shape)
    print("indices shape ", len(indices), "---", len(indices[0]))
    print("anchors shape ", len(anchors), "---", anchors[0].shape)
    print("tidxs shape ", len(tidxs), "---", tidxs[0].shape)
    print("xywhn shape ", len(xywhn), "---", xywhn[0].shape)
    # Losses
    for i, seg in enumerate(segments):  # layer index, layer predictions
      b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
      print("indices ",indices[i])
      tobj = torch.zeros(seg.shape[:4], dtype=seg.dtype, device=self.device)  # target obj
      n = b.shape[0]  # number of targets
      if n:
        pxy, pwh, _, pcls, pmask = seg[b, a, gj, gi].split((2, 2, 1, self.num_classes, nm), 1)  # subset of predictions
        # Box regression
        pxy = pxy.sigmoid() * 2 - 0.5
        pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
        pbox = torch.cat((pxy, pwh), 1)  # predicted box
        print("pxy shape ", pxy.shape)
        print("pwh shape ", pwh.shape)
        print("pbox shape ", pbox.shape)
        exit()
        iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
        lbox = lbox + (1.0 - iou).mean()  # iou loss
        # Objectness
        iou = iou.detach().clamp(0).type(tobj.dtype)
        if self.sort_obj_iou:
          j = iou.argsort()
          b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
        if self.gr < 1:
          iou = (1.0 - self.gr) + self.gr * iou
        tobj[b, a, gj, gi] = iou  # iou ratio
        # Classification
        if self.num_classes > 1:  # cls loss (only if multiple classes)
          t = torch.full_like(pcls, self.cn, device=self.device)  # targets
          t[range(n), tcls[i]] = self.cp
          lcls = lcls + self.BCEcls(pcls, t)  # BCE
        # Mask regression
        if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
          masks = F.interpolate(masks[None], (mask_h, mask_w), mode='nearest')[0]
        marea = xywhn[i][:, 2:].prod(1)  # mask width, height normalized
        mxyxy = xywh2xyxy(xywhn[i] * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=self.device))
        for bi in b.unique():
          j = b == bi  # matching index
          if self.overlap:
            mask_gti = torch.where(masks[bi][None] == tidxs[i][j].view(-1, 1, 1), 1.0, 0.0)
          else:
            mask_gti = masks[tidxs[i]][j]
          lseg = lseg + self.single_mask_loss(mask_gti, pmask[j], proto[bi], mxyxy[j], marea[j])
      obji = self.BCEobj(seg[..., 4], tobj)
      lobj = lobj + obji * self.balance[i]  # obj loss
    lbox = lbox * self.weight_box_loss
    lobj = lobj * self.weight_obj_loss
    lcls = lcls * self.weight_cls_loss
    lseg = lseg * self.weight_box_loss / bs
    loss = lbox + lobj + lcls + lseg
    loss = (loss * bs).requires_grad_(True)
    return loss, torch.cat((lbox, lseg, lobj, lcls)).detach()

  def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
    # Mask loss for one image
    pred_mask = (pred @ proto.view(self.num_masks, -1)).view(-1, *proto.shape[1:])  # (n,32) @ (32,160,160) -> (n,160,160)
    loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction='none')
    return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).mean()

  def build_targets(self, p, targets):
    # Build targets for compute_loss(), input targets(frame,class,x,y,w,h)
    na, nt = self.num_anchors, targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices, anch, tidxs, xywhn = [], [], [], [], [], []
    gain = torch.ones(8, device=self.device)  # normalized to gridspace gain
    ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    if self.overlap:
      batch = p[0].shape[0]
      ti = []
      for i in range(batch):
        num = (targets[:, 0] == i).sum()  # find number of targets of each image
        ti.append(torch.arange(num, device=self.device).float().view(1, num).repeat(na, 1) + 1)  # (na, num)
      ti = torch.cat(ti, 1)  # (na, nt)
    else:
      ti = torch.arange(nt, device=self.device).float().view(1, nt).repeat(na, 1)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None], ti[..., None]), 2)  # append anchor indices
    print("new targets ",targets.shape)
    g = 0.5  # bias
    off = torch.tensor(
      [
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],  # j,k,l,m
        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
      ],
      device=self.device).float() * g  # offsets
    for i in range(self.num_layers):
      anchors, shape = self.anchors[i], p[i].shape
      gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain
      # Match targets to anchors
      t = targets * gain  # shape(3,n,7)
      print("t0 ",t.shape)
      if nt:
        # Matches
        r = t[..., 4:6] / anchors[:, None]  # wh ratio
        j = torch.max(r, 1 / r).max(2)[0] < self.anchor_t  # compare
        # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
        t = t[j]  # filter
        # Offsets
        gxy = t[:, 2:4]  # grid xy
        gxi = gain[[2, 3]] - gxy  # inverse
        j, k = ((gxy % 1 < g) & (gxy > 1)).T
        l, m = ((gxi % 1 < g) & (gxi > 1)).T
        j = torch.stack((torch.ones_like(j), j, k, l, m))
        print("t1 ",t.shape)
        t = t.repeat((5, 1, 1))[j]
        print("t2 ",t.shape)
        offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
      else:
        t = targets[0]
        offsets = 0
      # Define
      print("t ",t.shape)
      bc, gxy, gwh, at = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
      (a, tidx), (b, c) = at.long().T, bc.long().T  # anchors, image, class
      gij = (gxy - offsets).long()
      gi, gj = gij.T  # grid indices
      # Append
      indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
      tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
      anch.append(anchors[a])  # anchors
      tcls.append(c)  # class
      tidxs.append(tidx)
      xywhn.append(torch.cat((gxy, gwh), 1) / gain[2:6])  # xywh normalized
    return tcls, tbox, indices, anch, tidxs, xywhn
import torch
import torch.nn as nn

class Conv(nn.Module):
  # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
  """
  Args:
    input_channel     (int):  The input channel number
      Default:  3
    output_channmel   (int):  The output channel number
      Default:  3
    kernel_size       (int):  The kernel size
      Default:  1
    stride            (int):  The stride
      Default:  1
    padding           (int):  The padding number
      Default:  1
    groups            (int):  The groups number
      Default:  1
    dilation          (int):  The dilation
      Default:  1
    act               (bool): The activation flag
      Default:  False
  """
  def __init__(self, input_channel=3, output_channmel=3, kernel_size=1, stride=1, padding=None, groups=1, dilation=1, act=True, device=torch.device("cpu")):
    super().__init__()
    self.default_act = nn.SiLU().to(device)  # default activation
    self.conv = nn.Conv2d(input_channel, output_channmel, kernel_size, stride, self.autopad(kernel_size, padding, dilation), groups=groups, dilation=dilation, bias=False, device=device)
    self.bn = nn.BatchNorm2d(output_channmel, device=device)
    self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity(device=device)
  def forward(self, x):
    """
    Args:
      x      (Tensor [bs, input_channel, H, W]):   The input tensor
    Returns:
      output (Tensor [bs, output_channel, (H + 2 * padding - kernel_size) / stride + 1, (W + 2 * padding - kernel_size) / stride + 1]): The output tensor
    """
    return self.act(self.bn(self.conv(x)))
  def forward_fuse(self, x):
    """
    Args:
      x      (Tensor [bs, input_channel, H, W]):   The input tensor
    Returns:
      output (Tensor [bs, output_channel, (H + 2 * padding - kernel_size) / stride + 1, (W + 2 * padding - kernel_size) / stride + 1]): The output tensor
    """
    return self.act(self.conv(x))
  def autopad(self, k=1, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    """
    Args:
      k (int): The kernel size
        Default: 1
      p (int): The padding number
        Default: None
      d (int): The dilation
        Default: 1
    Returns:
      padding (int): The padding number
    """
    if d > 1:
      k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
      p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Proto(nn.Module):
  # YOLOv5 mask Proto module for segmentation models
  """
  Args:
    input_channel (int):  The input channel number, equal to embed_dims
      Default:  256
    num_protos    (int):  The protos number
      Default:  256
    num_masks     (int):  The masks number
      Default:  32
  """
  def __init__(self, input_channel=256, num_protos=256, num_masks=32 ,device=torch.device("cpu")):  # ch_in, number of protos, number of masks
    super().__init__()
    self.cv1 = Conv(input_channel, num_protos, kernel_size=3, device=device)
    self.upsample = nn.Upsample(scale_factor=2, mode='nearest').to(device)
    self.cv2 = Conv(num_protos, num_protos, kernel_size=3, device=device)
    self.cv3 = Conv(num_protos, num_masks, device=device)
  def forward(self, x):
    """
    Args:
      x      (Tensor [bs, embed_dims, H, W]):   The input tensor
    Returns:
      output (Tensor [bs, num_masks, 2*H, 2*W]):  The output tensor
    """
    return self.cv3(self.cv2(self.upsample(self.cv1(x))))

class Detect(nn.Module):
  # YOLOv5 Detect head for detection models
  """
  Args:
    nc         (int):  The number of classes
      Default:  25
    cs         (int):  The coding size
      Default:  10
    anchors    (list of lists):  The anchor lists
      Default:  [[10,13, 16,30, 33,23], [10,13, 16,30, 33,23]]
    ch         (list of input channels number):  The channels numbers list for each layers
      Default:  [256,256]
  """
  def __init__(self, nc=25, cs=10, anchors=[[10,13, 16,30, 33,23], [10,13, 16,30, 33,23]], ch=[256,256] ,device=torch.device("cpu")):  # detection layer
    super().__init__()
    self.nc = nc  # number of classes
    self.no = nc + cs  # number of outputs per anchor
    self.nl = len(anchors)  # number of detection layers
    self.na = len(anchors[0]) // 2  # number of anchors
    self.register_buffer('anchors', torch.tensor(anchors).to(device).float().view(self.nl, -1, 2))  # shape(nl,na,2)
    self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1, device=device) for x in ch)  # output conv
  def forward(self, x):
    """
    Args:
      x      (list of Tensor [[bs, embed_dims, H, W],[bs, embed_dims, H, W]]):   The input tensor list
    Returns:
      output (list of Tensor [bs, num_anchors, H', W', code_size + num_classes]):  The output tensor list
    """
    for i in range(self.nl):
      x[i] = self.m[i](x[i])  # conv
      bs, _, ny, nx = x[i].shape  
      # x(bs,(nc+cs)*na,H,W) to x(bs,na,H,W,nc+cs)
      x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
    return x

class Segment(Detect):
  # YOLOv5 Segment head for segmentation models
  """
  Args:
    nc         (int):                            The number of classes
      Default:  25
    cs         (int):                            The coding size
      Default:  10
    anchors    (list of lists):                  The anchor lists
      Default:  [[10,13, 16,30, 33,23], [10,13, 16,30, 33,23]]
    nm          (int):                           The number of masks
      Default:  32
    npr         (int):                           The number of protos
      Default:  256
    ch         (list of input channels number):  The channels numbers list for each layers
      Default:  [256,256]
  """
  def __init__(self, nc=25, cs=10, anchors=[[10,13, 16,30, 33,23], [10,13, 16,30, 33,23]], nm=32, npr=256, ch=[256,256], device=torch.device("cpu")):
    super().__init__(nc, cs, anchors, ch)
    self.nm = nm  # number of masks
    self.npr = npr  # number of protos
    self.no = cs + nc + self.nm  # number of outputs per anchor
    self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1, device=device) for x in ch)  # output conv
    self.proto = Proto(ch[0], self.npr, self.nm, device=device)  # protos
    self.detect = Detect.forward
  def forward(self, x):
    p = self.proto(x[-1])
    x = self.detect(self, x)
    return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])
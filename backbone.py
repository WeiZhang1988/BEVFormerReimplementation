import torch
from torch import nn
import torch.nn.functional as F

class BackBone(nn.Module):
  """
  Args:
    stage_middle_channels     (int list): The stages' middle layer channels numbers
      Default: [64, 80, 96, 112]
    stage_out_channels        (int list): The stages' output channels numbers
      Default: [128, 256, 384, 512]
    num_block_per_stage       (int list): The number of block per stage
      Default: [1, 1, 2, 2]
    num_layer_per_block       (int):      The number of layer per block
      Default: 5
    -----Device-----
    device (torch.device): The device
      Default: cpu
  """
  def __init__(self,stage_middle_channels=[64, 80, 96, 112],stage_out_channels=[128, 256, 384, 512],num_block_per_stage=[1, 1, 2, 2],num_layer_per_block=5,device=torch.device("cpu")):
    super().__init__()
    self.stage_middle_channels  = stage_middle_channels
    self.stage_out_channels     = stage_out_channels
    self.num_block_per_stage    = num_block_per_stage
    self.num_layer_per_block    = num_layer_per_block
    self.device                 = device
    self.NN_vovnet              = VoVNet(stage_middle_channels,stage_out_channels,num_block_per_stage,num_layer_per_block,device)
  def forward(self,images):
    """
    Args:
      images    (Tensor [num_cams, bs, num_channels, height, width]): The input images
    Returns:
      Features  (Tensor [num_cams, bs, feature_dims]): The output features. The features dimension is the last of stage_out_channels
    """
    num_cams, bs, num_channels, height, width = images.size()
    features = self.NN_vovnet(images.view(num_cams * bs, num_channels, height, width)).view(num_cams, bs, -1)
    return features

class VoVNet(nn.Module):
  """
  Args:
    stage_middle_channels     (int list): The stages' middle layer channels numbers
      Default: [64, 80, 96, 112]
    stage_out_channels        (int list): The stages' output channels numbers
      Default: [128, 256, 384, 512]
    num_block_per_stage       (int list): The number of block per stage
      Default: [1, 1, 2, 2]
    num_layer_per_block       (int):      The number of layer per block
      Default: 5
    -----Device-----
    device (torch.device): The device
      Default: cpu
  """
  def __init__(self,stage_middle_channels=[64, 80, 96, 112],stage_out_channels=[128, 256, 384, 512],num_block_per_stage=[1, 1, 1, 1],num_layer_per_block=5,device=torch.device("cpu")):
    super().__init__()
    assert len(stage_middle_channels) == len(stage_out_channels) and len(stage_out_channels) == len(num_block_per_stage), "all list arguments must have same length"
    self.stage_middle_channels  = stage_middle_channels
    self.stage_out_channels     = stage_out_channels
    self.num_block_per_stage    = num_block_per_stage
    self.num_layer_per_block    = num_layer_per_block
    self.device                 = device
    self.NN_stem = nn.Sequential(Conv3x3(3,64,2).to(device),Conv3x3(64,64,1).to(device),Conv3x3(64,128,1).to(device))
    stem_out_channels = [128]
    stage_in_channels = stem_out_channels + stage_out_channels[:-1]
    osa_stages = []
    for i in range(len(stage_middle_channels)):
      osa_stages.append(OSA_Stage(stage_in_channels[i],stage_middle_channels[i],stage_out_channels[i],num_block_per_stage[i],num_layer_per_block,i+2,device))
    self.NN_stages = nn.Sequential(*osa_stages)
  def forward(self,images):
    """
    Args:
      images    (Tensor [bs, num_channels, height, width]):                           The input images
    Returns:
      Features  (tensor [bs, stage_out_channels[-1], height/2/2/2/2, width/2/2/2/2]): The output features. The features dimension is the last of stage_out_channels
    """
    # images      [bs, num_channels,            height,         width]
    # ->features  [bs, 128(fixed in the impl.), height/2,       width/2]
    # --------->  [bs, stage_out_channels[-1],  height/2/2/2/2, width/2/2/2/2]
    features = self.NN_stem(images)
    features = self.NN_stages(features)
    #features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
    return features

class OSA_Stage(nn.Module):
  """
  Args:
    in_channels     (int): The input channels number
    middle_channels (int): The middle layer channels number
    out_channels    (int): The output channels number
    num_block       (int): The number of OSA blocks
    num_layer       (int): The number of layers
    nth_stage       (int): The stage number that this OSA stage belongs to
    -----Device-----
    device (torch.device): The device
      Default: cpu
  """
  def __init__(self,in_channels,middle_channels,out_channels,num_block,num_layer,nth_stage,device=torch.device("cpu")):
    super().__init__()
    self.in_channels     = in_channels
    self.middle_channels = middle_channels
    self.out_channels    = out_channels
    self.num_layer       = num_layer
    self.num_block       = num_block
    self.nth_stage       = nth_stage
    self.device          = device
    assert nth_stage>1, "add stem stage before OSA stage"
    osa_modules = []
    if nth_stage > 2:
      osa_modules.append(nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True))
    osa_modules.append(OSA_Block(in_channels,middle_channels,out_channels,num_layer,False,device))
    for _ in range(num_block-1):
      osa_modules.append(OSA_Block(out_channels,middle_channels,out_channels,num_layer,True,device))
    self.NN_stage = nn.Sequential(*osa_modules)
  def forward(self,x):
    return self.NN_stage(x)


class OSA_Block(nn.Module):
  """
  Args:
    in_channels               (int):  The input channels number
    middle_channels           (int):  The middle layer channels number
    out_channels              (int):  The out channels number
    num_layer                 (int):  The number of layers
    in_out_channels_identity  (bool): Whether input and output have the same channels number
      Default: False
    -----Device-----
    device (torch.device): The device
      Default: cpu
  """
  def __init__(self,in_channels,middle_channels,out_channels,num_layer,in_out_channels_identity=False,device=torch.device("cpu")):
    super().__init__()
    self.in_channels              = in_channels
    self.middle_channels          = middle_channels
    self.out_channels             = out_channels
    self.num_layer                = num_layer
    self.in_out_channels_identity = in_out_channels_identity
    self.device                   = device
    middle_in_channels = in_channels
    self.NN_layers = nn.ModuleList()
    for _ in range(num_layer):
      self.NN_layers.append(Conv3x3(middle_in_channels,middle_channels).to(device))
      middle_in_channels = middle_channels
    # feature aggregation
    middle_in_channels = in_channels + num_layer * middle_channels
    self.NN_Concat = Conv1x1(middle_in_channels,out_channels).to(device)
  def forward(self,x):
    residual = x
    output = []
    output.append(x)
    for NN_layer in self.NN_layers:
      x = NN_layer(x)
      output.append(x)
    x = torch.cat(output, dim=1)
    x = self.NN_Concat(x)
    if self.in_out_channels_identity:
      return x + residual
    else:
      return x

class Conv3x3(nn.Module):
  """
  Args:
    in_channels  (int): The input channels number
    out_channels (int): The output channels number
    stride       (int): The stride number
      Default: 1
    groups       (int): The group number
      Default: 1
    padding      (int): The padding
      Default: 1
  """
  def __init__(self,in_channels,out_channels,stride=1,groups=1,padding=1):
    super().__init__()
    self.in_channels    = in_channels
    self.out_channels   = out_channels
    self.stride         = stride
    self.groups         = groups
    self.padding        = padding
    self.NN_Conv3x3     = nn.Sequential(
                          nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=padding,groups=groups,bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(inplace=True))
  def forward(self,x):
    return self.NN_Conv3x3(x)

class Conv1x1(nn.Module):
  """
  Args:
    in_channels  (int): The input channels number
    out_channels (int): The output channels number
    stride       (int): The stride number
      Default: 1
    groups       (int): The group number
      Default: 1
    padding      (int): The padding
      Default: 1
  """
  def __init__(self,in_channels,out_channels,stride=1,groups=1,padding=0):
    super().__init__()
    self.in_channels    = in_channels
    self.out_channels   = out_channels
    self.stride         = stride
    self.groups         = groups
    self.padding        = padding
    self.NN_Conv1x1 = nn.Sequential(
                      nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,padding=padding,groups=groups,bias=False),
                      nn.BatchNorm2d(out_channels),
                      nn.ReLU(inplace=True))
  def forward(self,x):
    return self.NN_Conv1x1(x)
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, einsum

class SpatialCrossAttention(nn.Module):
  """
  Args:
    num_cams      (int):   The number of cameras
      Default: 6
    num_zAnchors  (int):   The number of z anchors
      Default: 4
    dropout       (float): The drop out rate
      Default: 0.1
    embed_dims    (int):   The embedding dimension of attention. The same as inputs embedding dimension.
      Default: 256   
    num_heads     (int):   The number of heads.
      Default: 8   
    num_levels    (int):   The number of scale levels in a single sequence
      Default: 4   
    num_points    (int):   The number of sampling points in a single level
      Default: 2
    -----Device-----
    device (torch.device): The device
      Default: cpu
  """
  def __init__(self,num_cams=6,num_zAnchors=4,dropout=0.1,embed_dims=256,num_heads=8,num_levels=4,num_points=2,device=torch.device("cpu")):
    super().__init__()
    self.num_cams      = num_cams
    self.num_zAnchors  = num_zAnchors
    self.dropout       = dropout
    self.embed_dims    = embed_dims
    self.num_heads     = num_heads
    self.num_levels    = num_levels
    self.num_points    = num_points
    self.device        = device
    self.xy            = 2
    assert self.embed_dims % self.num_heads == 0, "embedding dimension must be divisible by number of heads"
    self.head_dims     = self.embed_dims // self.num_heads
    self.NN_to_Q             = nn.Linear(self.embed_dims,self.embed_dims).to(device)
    self.NN_to_K             = nn.Linear(self.embed_dims,self.embed_dims).to(device)
    self.NN_to_V             = nn.Linear(self.embed_dims,self.embed_dims).to(device)
    self.NN_sampling_offsets = nn.Linear(self.embed_dims, self.num_heads * self.num_levels * self.num_points * self.num_zAnchors * 2).to(device)
    self.NN_output           = nn.Linear(self.embed_dims, self.embed_dims).to(device)
    self.NN_dropout          = nn.Dropout(dropout).to(device)
  def forward(self,query,key,value,reference_points=None,spatial_shapes=None,reference_points_cam=None,bev_mask=None):
    """
    Args:
      query                (Tensor [bs, num_query, embed_dims]):              The query
      key                  (Tensor [num_cams, bs, num_key, embed_dims]):      The key
      value                (Tensor [num_cams, bs, num_value, embed_dims]):    The value
      reference_points     (Tensor [bs, num_query, 4]):                       The normalized reference points, 4 means x y z 1. Actually not used here
      spatial_shapes       (Tensor [num_levels, 2]):                          The spatial shape of features in different levels
      reference_points_cam (Tensor [num_cam, bs, num_query, num_zAnchor, 2]): The image pixel ratio projected from reference points to each camera
      bev_mask             (Tensor [num_cam, bs, num_query, num_zAnchor]):    Which of reference_points_cam is valid
    Returns:
      Attention            (tensor [bs, num_query, embed_dims])
    """
    # residual [bs, num_query, embed_dims]
    residual = query
    # slots [bs, num_query, embed_dims]
    slots = torch.zeros_like(query)
    bs_q, num_query, _      = query.size()
    num_cams,bs_k,num_key,_ = key.size()
    _,_,num_value,_         = value.size()
    assert bs_q     == bs_k,          "batch size must be equal"
    assert num_cams == self.num_cams, "camera numbers must matched"
    assert num_key  == num_value,     "number of value must equals to number of key"
    bs = bs_q
    num_levels = reference_points_cam.size(3)
    indices = []
    # traverse all cameras:
    #   sum up and squeeze in last dimension from first batch
    #   append
    for i, mask_per_img in enumerate(bev_mask):
      # index_query_per_img [num_query]
      index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
      indices.append(index_query_per_img)
    max_len = max([len(each) for each in indices])
    # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
    # query_rebatch [bs, num_cams, max_len, embed_dims]
    query_rebatch = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
    # reference_points_rebatch [bs, num_cams, max_len, num_zAnchor, 2]
    reference_points_rebatch = reference_points_cam.new_zeros([bs, self.num_cams, max_len, self.num_zAnchors, 2])
    # traverse all cameras:
    #   repeat query and reference points to all cameras
    for j in range(bs):
      for i, reference_points_per_img in enumerate(reference_points_cam): 
        index_query_per_img = indices[i]
        query_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
        reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
    # query_rebatch [bs, num_cams,  max_len, embed_dims]
    # --------------> [bs * num_cams, max_len, embed_dims]
    query_rebatch = query_rebatch.view(bs*self.num_cams, max_len, self.embed_dims)
    # reference_points_rebatch [bs, num_cams,  max_len, num_levels, 2]
    # -----------------------> [bs * num_cams, max_len, num_levels, 2]
    reference_points_rebatch = reference_points_rebatch.view(bs*self.num_cams, max_len, num_levels, 2)
    # key [num_cams, bs,  num_key, embed_dims]
    # --> [bs * num_cams, num_key, embed_dims]
    key = key.reshape(bs*self.num_cams,num_key,self.embed_dims).contiguous()
    # value [num_cams, bs,  num_key, embed_dims]
    # ----> [bs * num_cams, num_key, embed_dims]
    value = value.reshape(bs*self.num_cams,num_value,self.embed_dims).contiguous()
    #####################################################################################################################################################################
    # start of multi-scale deformable attention computation #############################################################################################################
    #####################################################################################################################################################################
    bc, nq, _ = query_rebatch.shape
    bc, nk, _ = key.shape
    bc, nv, _ = value.shape
    # sampling_offsets [bs * num_cams, nq, num_heads, num_levels, num_points * num_zAnchors, 2]
    sampling_offsets = self.NN_sampling_offsets(query_rebatch).view(bc, nq, self.num_heads, self.num_levels, self.num_points * self.num_zAnchors, self.xy)
    assert nk == self.num_levels * self.num_points * self.num_zAnchors,   "total key number does not match numbers of levels, points and z anchors"
    assert nv == self.num_levels * self.num_points * self.num_zAnchors,   "total value number does not match numbers of levels, points and z anchors"
    #assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
    q = self.NN_to_Q(query_rebatch) 
    k = self.NN_to_K(key)    
    v = self.NN_to_V(value) 
    # q  [bs * num_cams, nq, embed_dims]
    # -> [bs * num_cams, nq, num_heads, head_dims]
    q = rearrange(q, 'b q (h d) -> b q h d', h=self.num_heads) 
    # k  [bs * num_cams, nk, embed_dims]
    # -> [bs * num_cams, nk, num_heads, head_dims]
    k = rearrange(k, 'b k (h d) -> b k h d', h=self.num_heads)
    # v  [bs * num_cams, nv, embed_dims]
    # -> [bs * num_cams, nv, num_heads, head_dims]
    v = rearrange(v, 'b v (h d) -> b v h d', h=self.num_heads)
    # attention_weights [bs * num_cams, nq, num_heads, num_levels * num_points * num_zAnchors]
    # ----------------> [bs * num_cams, nq, num_heads, num_levels,  num_points * num_zAnchors]
    attention_weights = einsum(q, k, 'b q h d, b k h d -> b q h k')
    attention_weights = attention_weights.softmax(-1)
    attention_weights = rearrange(attention_weights, 'b q h (l pz) -> b q h l pz', l=self.num_levels, pz=self.num_points * self.num_zAnchors)
    assert reference_points_rebatch.shape[1]  == nq, "second dim of reference_points must equal to nq"
    assert reference_points_rebatch.shape[-1] == 2,  "last dim of reference_points must be 2"
    """
    For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
    After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
    For each referent point, we sample `num_points` sampling points.
    For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
    """
    # offset_normalizer [num_levels, 2]
    offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
    # reference_points_rebatch [bs * num_cams, nq, 1(to extend to num_heads), 1(to extend to num_levels), 1(to extend to num_points), num_zAnchor, 2]
    reference_points_rebatch = reference_points_rebatch[:, :, None, None, None, :, :]
    # sampling_offsets [bs * num_cams, nq, num_heads, num_levels, num_points * num_zAnchors, 2]
    # ---------------> [bs * num_cams, nq, num_heads, num_levels, num_points,  num_zAnchors, 2]
    sampling_offsets = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
    sampling_offsets = sampling_offsets.view(bs * num_cams, nq, self.num_heads, self.num_levels, self.num_points, self.num_zAnchors, self.xy)
    # sampling_locations [bs * num_cams, nq, num_heads, num_levels, num_points,  num_zAnchors, 2]
    # -----------------> [bs * num_cams, nq, num_heads, num_levels, num_points * num_zAnchors, 2]
    sampling_locations = reference_points_rebatch + sampling_offsets
    sampling_locations = sampling_locations.view(bs * num_cams, nq, self.num_heads, self.num_levels, self.num_points * self.num_zAnchors, self.xy)
    """
    multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights)
    """
    assert tuple(v.shape) == (bc, nv, self.num_heads, self.head_dims)
    assert tuple(spatial_shapes.shape) == (self.num_levels, self.xy)
    assert tuple(sampling_locations.shape) == (bc, nq, self.num_heads, self.num_levels, self.num_points * self.num_zAnchors, self.xy)
    assert tuple(attention_weights.shape) ==  (bc, nq, self.num_heads, self.num_levels, self.num_points * self.num_zAnchors)
    tmp = [int(H_.item()) * int(W_.item()) for H_, W_ in spatial_shapes]
    value_list = v.split(tmp, dim=1)
    # sampling_grids [bs * num_cams, nq, num_heads, num_levels, num_points * num_zAnchors, 2]
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(spatial_shapes):
      # value_l_ [bs * num_cams,             H_ * W_,                num_heads,              head_dims] 
      # -------> [bs * num_cams,             H_ * W_,                num_heads * head_dims]
      # -------> [bs * num_cams,             num_heads * head_dims, H_ * W_]
      # -------> [bs * num_cams * num_heads, head_dims,             H_,                     W_]
      value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * self.num_cams * self.num_heads, self.head_dims, int(H_.item()), int(W_.item()))
      # sampling_grid_l_  [bs * num_cams,              nq,         num_heads,                   num_points * num_zAnchors, 2]
      # --------------->  [bs * num_cams,              num_heads,  nq,                          num_points * num_zAnchors, 2]
      # --------------->  [bs * num_cams * num_heads,  nq,         num_points * num_zAnchors,   2]
      sampling_grid_l_ = sampling_grids[:, :, :,level].transpose(1, 2).flatten(0, 1)
      # sampling_value_l_ [bs * num_cams * num_heads,  head_dims,  nq,  num_points * num_zAnchors]
      sampling_value_l_ = F.grid_sample(
        value_l_,
        sampling_grid_l_,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False)
      sampling_value_list.append(sampling_value_l_)
    # attention_weights [bs * num_cams,              nq,                        num_heads,   num_levels,  num_points * num_zAnchors]
    # ----------------> [bs * num_cams,              num_heads,                 nq,          num_levels,  num_points * num_zAnchors]
    # ----------------> [bs * num_cams * num_heads,  1(to extend to head_dims), num_queries, num_levels * num_points * num_zAnchors]
    attention_weights = attention_weights.transpose(1, 2).reshape(bc * self.num_heads, 1, num_query, self.num_levels * self.num_points * self.num_zAnchors)
    # sampling_value_list [[bs * num_cams * num_heads, head_dims, nq, num_points * num_zAnchors]....]
    # ------------------>  [bs * num_cams * num_heads, head_dims, nq, num_levels(get the dimension by stack in dim -2), num_points * num_zAnchors]
    # ------------------>  [bs * num_cams * num_heads, head_dims, nq, num_levels * num_points * num_zAnchors(by flatten)]
    # ------------------>  [bs * num_cams * num_heads, head_dims, nq, 1(by sum after elementwise multiplication)]
    # ------------------>  [bs * num_cams, num_heads * head_dims(==embed_dims), nq]
    # ------------------>  [bs * num_cams, nq, embed_dims]
    attention = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(bc, self.num_heads * self.head_dims, nq).transpose(1, 2).contiguous()
    #####################################################################################################################################################################
    # end of multi-scale deformable attention computation ###############################################################################################################
    #####################################################################################################################################################################
    # attention [bs * num_cams, max_len, embed_dims]
    # --------> [bs,  num_cams, max_len, embed_dims]
    attention = attention.view(bs, self.num_cams, max_len, self.embed_dims)
    # traverse all cameras:
    #   fill positions with zeros to match shape
    for j in range(bs):
      for i, index_query_per_img in enumerate(indices):
        slots[j, index_query_per_img] += attention[j, i, :len(index_query_per_img)]
    # bev_mask [num_cam, bs, num_query, num_levels]
    # -->count [num_cam, bs, num_query]
    # -------> [bs, num_query]
    count = bev_mask.sum(-1) > 0
    count = count.permute(1, 2, 0).sum(-1)
    count = torch.clamp(count, min=1.0)
    # slots [bs, num_query, embed_dims]
    # count[..., None] makes count [bs, num_query, 1(to extend to embed_dims)]
    slots = slots / count[..., None]
    slots = self.NN_output(slots)
    # return spatial attention [bs, num_query, embed_dims]
    return self.NN_dropout(slots) + residual



class TemporalSelfAttention(nn.Module):
  """
  Args:
    num_sequences (int):   The number of sequences of queries, kies and values.
      Default: 2
    dropout       (float): The drop out rate
      Default: 0.1
    embed_dims    (int):   The embedding dimension of attention. The same as inputs embedding dimension.
      Default: 256   
    num_heads     (int):   The number of heads.
      Default: 8   
    num_levels    (int):   The number of scale levels in a single sequence
      Default: 4   
    num_points    (int):   The number of sampling points in a single level
      Default: 4
    -----Device-----
    device (torch.device): The device
      Default: cpu
  -------------------------------------------------------------------------------------------------
  """
  def __init__(self,num_sequences=2,dropout=0.1,embed_dims=256,num_heads=8,num_levels=4,num_points=4,device=torch.device("cpu")):
    super().__init__()
    self.num_sequences = num_sequences
    self.dropout       = dropout
    self.embed_dims    = embed_dims
    self.num_heads     = num_heads
    self.num_levels    = num_levels
    self.num_points    = num_points
    self.device        = device
    self.xy            = 2
    assert self.embed_dims % self.num_heads == 0, "embedding dimension must be divisible by number of heads"
    self.head_dims      = self.embed_dims // self.num_heads
    self.NN_to_Q             = nn.Linear(self.embed_dims,self.embed_dims).to(device)
    self.NN_to_K             = nn.Linear(self.embed_dims,self.embed_dims).to(device)
    self.NN_to_V             = nn.Linear(self.embed_dims,self.embed_dims).to(device)
    self.NN_sampling_offsets = nn.Linear(self.embed_dims, self.num_heads * self.num_levels * self.num_points * 2).to(device)
    self.NN_output           = nn.Linear(self.embed_dims, self.embed_dims).to(device)
    self.NN_dropout          = nn.Dropout(dropout).to(device)
    assert self.num_sequences>0, "value length must be larger than zero"
  def forward(self,query,key_hist=[],value_hist=[],reference_points=None,spatial_shapes=None):
    """
    Args:
      query             (Tensor         [bs, num_query, embed_dims]):    The query
      key_hist          (list of Tensor [bs, num_key,   embed_dims]s):   The key history. num_key should equel to num_levels * num_points
      value_hist        (list of Tensor [bs, num_value, embed_dims]s):   The value history
      reference_points  (Tensor         [bs, num_query, num_levels, 2]): The normalized reference points. Passed though to multi scale deformable attention layer
      spatial_shapes    (Tensor         [num_levels, 2]):                The Spatial shape of features in different levels. Passed though to multi scale deformable attention layer
    Returns:
      Attention         (tensor         [bs, num_query, embed_dims])
    """
    # residual [bs, num_query, embed_dims]
    residual   = query
    query_list = [query for _ in range(self.num_sequences)] 
    key_list   = key_hist
    value_list = value_hist
    key_list.insert(0,query)
    value_list.insert(0,query)
    bs, num_query, _ = query.shape
    bs, num_key,   _ = key_list[0].shape
    bs, num_value, _ = value_list[0].shape
    assert len(query_list) == len(key_list) and len(key_list) == len(value_list) and len(value_list) == self.num_sequences, "length of query_list, key_list and value_list must equal to num_sequences"
    # queries [bs, num_sequences * num_query, embed_dims]
    # ------> [bs * num_sequences, num_query, embed_dims]
    queries = torch.cat(query_list,dim=1).view(bs * self.num_sequences, num_query, self.embed_dims)
    # keies   [bs, num_sequences * num_key,   embed_dims]
    # ------> [bs * num_sequences, num_key,   embed_dims]
    keies = torch.cat(key_list,dim=1).view(bs * self.num_sequences, num_key, self.embed_dims)
    # values  [bs, num_sequences * num_value, embed_dims]
    # ------> [bs * num_sequences, num_value, embed_dims]
    values = torch.cat(value_list,dim=1).view(bs * self.num_sequences, num_value, self.embed_dims)
    assert num_key   == self.num_levels * self.num_points,   "total key number does not match numbers of levels and points"
    assert num_value == self.num_levels * self.num_points,   "total value number does not match numbers of levels and points"
    # reference_points [bs, num_query,                 num_levels, 2]
    # ---------------> [bs * num_sequences, num_query, num_levels, 2]
    reference_points_list = [reference_points for _ in range(self.num_sequences)] 
    reference_points = torch.cat(reference_points_list,dim=0).view(bs * self.num_sequences, num_query, self.num_levels,2)
    #####################################################################################################################################################################
    # start of attention computation ####################################################################################################################################
    #####################################################################################################################################################################
    # sampling_offsets [bs * num_sequences, num_query, num_heads, num_levels, num_points, 2]
    sampling_offsets = self.NN_sampling_offsets(queries).view(bs * self.num_sequences, num_query, self.num_heads, self.num_levels, self.num_points, self.xy)
    assert num_key   == self.num_levels * self.num_points,   "total key number does not match numbers of levels and points"
    assert num_value == self.num_levels * self.num_points,   "total value number does not match numbers of levels and points"
    #assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
    q = self.NN_to_Q(queries) 
    k = self.NN_to_K(keies)    
    v = self.NN_to_V(values)  
    # q  [bs * num_cams, nq, embed_dims]
    # -> [bs * num_cams, nq, num_heads, head_dims]
    q = rearrange(q, 'b q (h d) -> b q h d', h=self.num_heads) 
    # k  [bs * num_cams, nk, embed_dims]
    # -> [bs * num_cams, nk, num_heads, head_dims]
    k = rearrange(k, 'b k (h d) -> b k h d', h=self.num_heads)
    # v  [bs * num_cams, nv, embed_dims]
    # -> [bs * num_cams, nv, num_heads, head_dims]
    v = rearrange(v, 'b v (h d) -> b v h d', h=self.num_heads)
    # attention_weights [bs * num_sequences, num_query, num_heads, num_levels * num_points]
    # ----------------> [bs * num_sequences, num_query, num_heads, num_levels,  num_points]
    attention_weights = einsum(q, k, 'b q h d, b k h d -> b q h k')
    attention_weights = attention_weights.softmax(-1)
    attention_weights = rearrange(attention_weights, 'b q h (l p) -> b q h l p', l=self.num_levels)
    assert reference_points.shape[1]  == num_query, "second dim of reference_points must equal to num_query"
    assert reference_points.shape[-1] == 2,         "last dim of reference_points must be 2"
    # offset_normalizer [num_levels, 2]
    offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
    # reference_points  extends to [bs * num_sequences, num_query, 1(to extend to num_heads), num_levels, 1(to extend to num_points), 2]
    # offset_normalizer extends to [bs * num_sequences, num_query, num_heads, num_levels, num_points, 2]
    sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
    """
    multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights)
    """
    assert tuple(v.shape)                  == (bs * self.num_sequences, num_value, self.num_heads, self.head_dims)
    assert tuple(spatial_shapes.shape)     == (self.num_levels, self.xy)
    assert tuple(sampling_locations.shape) == (bs * self.num_sequences, num_query, self.num_heads, self.num_levels, self.num_points, self.xy)
    assert tuple(attention_weights.shape)  == (bs * self.num_sequences, num_query, self.num_heads, self.num_levels, self.num_points)
    tmp = [int(H_.item()) * int(W_.item()) for H_, W_ in spatial_shapes]
    value_list = v.split(tmp, dim=1)
    # sampling_grids [bs * num_sequences, num_query, num_heads, num_levels, num_points, 2]
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(spatial_shapes):
      # value_l_ [bs * num_sequences,             H_ * W_,                num_heads,              head_dims] 
      # -------> [bs * num_sequences,             H_ * W_,                num_heads * head_dims]
      # -------> [bs * num_sequences,             num_heads * head_dims,  H_ * W_]
      # -------> [bs * num_sequences * num_heads, head_dims,              H_,                     W_]
      value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * self.num_sequences * self.num_heads, self.head_dims, int(H_.item()), int(W_.item()))
      # sampling_grid_l_ [bs * num_sequences,              num_query, num_heads,                   num_points, 2]
      # ---------------> [bs * num_sequences,              num_heads,                   num_query, num_points, 2]
      # ---------------> [bs * num_sequences * num_heads,  num_query, num_points,                  2]
      sampling_grid_l_ = sampling_grids[:, :, :,level].transpose(1, 2).flatten(0, 1)
      # sampling_value_l_ [bs * num_sequences * num_heads, head_dims, num_queries, num_points]
      sampling_value_l_ = F.grid_sample(
        value_l_,
        sampling_grid_l_,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False)
      sampling_value_list.append(sampling_value_l_)
    # attention_weights [bs * num_sequences, num_query,   num_heads,   num_levels,   num_points]
    # ----------------> [bs * num_sequences, num_heads,   num_query,   num_levels,   num_points]
    # ----------------> [bs * num_sequences * num_heads,  1(to extend to head_dims), num_query, num_levels * num_points]
    attention_weights = attention_weights.transpose(1, 2).reshape(bs * self.num_sequences * self.num_heads, 1, num_query, self.num_levels * self.num_points)
    # sampling_value_list [[bs * num_sequences * num_heads, head_dims, num_query, num_points]....]
    # ------------------>  [bs * num_sequences * num_heads, head_dims, num_query, num_levels(get the dimension by stack in dim -2), num_points]
    # ------------------>  [bs * num_sequences * num_heads, head_dims, num_query, num_levels * num_points(by flatten)]
    # ------------------>  [bs * num_sequences * num_heads, head_dims, num_query, 1(by sum after elementwise multiplication)]
    # ------------------>  [bs * num_sequences, num_heads * head_dims(==embed_dims), num_query]
    # ------------------>  [bs * num_sequences, num_query, embed_dims]
    attention = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(bs * self.num_sequences, self.num_heads * self.head_dims, num_query).transpose(1, 2).contiguous()
    #####################################################################################################################################################################
    # end of attention computation ######################################################################################################################################
    #####################################################################################################################################################################
    # attention [bs * num_sequences, num_query, embed_dims]
    # --------> [num_query, embed_dims, bs * num_sequences]
    # --------> [num_query, embed_dims, bs,  num_sequences]
    # --------> [num_query, embed_dims, bs]
    # --------> [bs, num_query embed_dims]
    attention = attention.permute(1, 2, 0).view(num_query, self.embed_dims, bs, self.num_sequences).mean(-1).permute(2, 0, 1)
    # return temporal attention [bs, num_query, embed_dims]
    return self.NN_dropout(self.NN_output(attention)) + residual

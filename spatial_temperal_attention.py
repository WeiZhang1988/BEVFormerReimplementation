import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, einsum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpatialCrossAttention(nn.Module):
  """
  Args:
    num_cams (int): The number of cameras
      Default: 6
    dropout (float): A dropout layer parameter
      Default: 0.
    deformable_attention (nn): as name suggests
  """
  def __init__(self,num_cams=6,dropout=0.1,deformable_attention=None):
    super().__init__()
    self.num_cams = num_cams
    self.dropout = nn.Dropout(dropout)
    self.deformable_attention = deformable_attention
    assert deformable_attention is not None, "deformable_attention must exists"
    self.embed_dims = deformable_attention.embed_dims
    self.output_proj = nn.Linear(self.embed_dims, self.embed_dims)
  def forward(self,query,key,value,reference_points=None,spatial_shapes=None,reference_points_cam=None,bev_mask=None):
    """
    Args:
      query (Tensor): query of transformer with shape (bs, num_query, embed_dims)
      key (Tensor): key of transformer with shape (num_cams, bs, num_key, embed_dims)
      value (Tensor): value of transformer with shape (num_cams, bs, num_value, embed_dims)
      reference_points (Tensor): the normalized reference points with shape (bs, num_query, 4)
      spatial_shapes (Tensor): spatial shape of features in different levels. With shape (num_levels, 2)
      reference_points_cam (Tensor): Image pixel ratio projected from reference points 
        to each camera. With shape(num_cam, bs, num_query, num_levels, 2)
      bev_mask (Tensor): which of reference_points_cam is valid. 
        With shape(num_cam, bs, num_query, num_levels)
    Returns:
      tensor with shape (bs, num_query, embed_dims)
    """
    residual = query
    slots = torch.zeros_like(query)
    bs_q, num_query, _ = query.size()
    num_cams,bs_k,num_key,_ = key.size()
    _,_,num_value,_ = value.size()
    assert bs_q == bs_k, "batch size must be equal"
    assert num_cams == self.num_cams, "camera numbers must matched"
    assert num_key == num_value, "number of value must equals to number of key"
    bs = bs_q
    num_levels = reference_points_cam.size(3)
    indices = []
    for i, mask_per_img in enumerate(bev_mask):
      index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
      indices.append(index_query_per_img)
    max_len = max([len(each) for each in indices])
    # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
    queries_rebatch = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
    reference_points_rebatch = reference_points_cam.new_zeros([bs, self.num_cams, max_len, num_levels, 2])
    for j in range(bs):
      for i, reference_points_per_img in enumerate(reference_points_cam):   
        index_query_per_img = indices[i]
        queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
        reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
    key = key.reshape(bs*self.num_cams,num_key,self.embed_dims)
    value = value.reshape(bs*self.num_cams,num_value,self.embed_dims)
    attention = self.deformable_attention(query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims),key=key,value=value,reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, num_levels, 2),spatial_shapes=spatial_shapes).view(bs, self.num_cams, max_len, self.embed_dims)
    for j in range(bs):
      for i, index_query_per_img in enumerate(indices):
        slots[j, index_query_per_img] += attention[j, i, :len(index_query_per_img)]
    count = bev_mask.sum(-1) > 0
    count = count.permute(1, 2, 0).sum(-1)
    count = torch.clamp(count, min=1.0)
    slots = slots / count[..., None]
    slots = self.output_proj(slots)
    return self.dropout(slots) + residual

class MSDeformableAttention3D(nn.Module):
  """
  Args:
    embed_dims (int): The embedding dimension of attention. The same as inputs embedding dimension.
      Default: 256
    num_heads (int):  The number of heads.
      Default: 8
    num_levels (int): The number of scale levels
      Default: 4
    num_points (int): The number of sampling points
      Default: 4
  """
  def __init__(self,embed_dims=256,num_heads=8,num_levels=4,num_points=4):
    super().__init__()
    self.embed_dims = embed_dims
    self.num_heads  = num_heads
    self.num_levels = num_levels
    self.num_points = num_points
    self.xy         = 2
    assert self.embed_dims % self.num_heads == 0, "embedding dimension must be divisible by number of heads"
    self.head_dims  = self.embed_dims // self.num_heads
    self.to_Q = nn.Linear(self.embed_dims,self.embed_dims)
    self.to_K = nn.Linear(self.embed_dims,self.embed_dims)
    self.to_V = nn.Linear(self.embed_dims,self.embed_dims)
    self.sampling_offsets = nn.Linear(self.embed_dims, self.num_heads * self.num_levels * self.num_points * 2)
  def forward(self,query,key,value,reference_points=None,spatial_shapes=None):
    """
    Args:
      query (Tensor): Query of transformer with shape (bs, num_query, embed_dims)
      key (Tensor): Key of transformer with shape (bs, num_key, embed_dims). num_key should equel to num_levels * num_points
      value (Tensor): Value of transformer with shape (bs, num_value, embed_dims)
      reference_points (Tensor):  The normalized reference points with shape (bs, num_query, num_levels, 2),
        all elements are ranged in [0, 1], top-left (0,0),
        bottom-right (1, 1), including padding area.
        or (N, Length_{query}, num_levels, 4), add
        additional two dimensions is (w, h) to
        form reference boxes.
      spatial_shapes (Tensor): Spatial shape of features in different levels. With shape (num_levels, 2), last dimension represents (h, w).
    Returns:
      tensor with shape (bs, num_query, embed_dims)
    """
    bs, num_query, _ = query.shape
    bs, num_key,   _ = key.shape
    bs, num_value, _ = value.shape
    sampling_offsets = self.sampling_offsets(query).view(bs, num_query, self.num_heads, self.num_points, self.num_levels, self.xy)
    assert num_key   == self.num_levels * self.num_points,   "total key number does not match numbers of levels and points"
    assert num_value == self.num_levels * self.num_points,   "total value number does not match numbers of levels and points"
    #assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
    q = self.to_Q(query) 
    k = self.to_K(key)    
    v = self.to_V(value)  
    q = rearrange(query, 'b q (h d) -> b q h d', h=self.num_heads)#q shape [b, num_query, num_heads, head_dims] 
    k = rearrange(key,   'b k (h d) -> b k h d', h=self.num_heads)#k shape [b, num_key,   num_heads, head_dims]
    v = rearrange(value, 'b v (h d) -> b v h d', h=self.num_heads)#v shape [b, num_value, num_heads, head_dims]
    attention_weights = einsum(q, k, 'b q h d, b k h d -> b q h k')
    print("attention_weights shape ",attention_weights.shape) 
    attention_weights = attention_weights.softmax(-1)
    attention_weights = rearrange(attention_weights, 'b q h (l p) -> b q h l p', l=self.num_levels)
    assert reference_points.shape[1]  == num_query, "second dim of reference_points must equal to num_query"
    assert reference_points.shape[-1] == 2,         "last dim of reference_points must be 2"
    """
    For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
    After proejcting, each BEV query has `num_Z_anchors == num_levels` reference points in each 2D image.
    For each referent point, we sample `num_points` sampling points.
    For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
    """
    offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
    reference_points = reference_points[:, :, None, None, :, :]
    sampling_offsets = sampling_offsets / offset_normalizer[None, None, None, None, :, :]
    sampling_offsets = sampling_offsets.view(bs, num_query, self.num_heads, self.num_points, self.num_levels, self.xy)
    sampling_locations = reference_points + sampling_offsets
    sampling_locations = sampling_locations.view(bs, num_query, self.num_heads, self.num_levels, self.num_points, self.xy)
    """
    multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights)
    """
    assert tuple(v.shape) == (bs, num_value, self.num_heads, self.head_dims)
    assert tuple(spatial_shapes.shape) == (self.num_levels, self.xy)
    assert tuple(sampling_locations.shape) == (bs, num_query, self.num_heads, self.num_levels, self.num_points, self.xy)
    assert tuple(attention_weights.shape) == (bs, num_query, self.num_heads, self.num_levels, self.num_points)
    tmp = [int(H_.item()) * int(W_.item()) for H_, W_ in spatial_shapes]
    value_list = v.split(tmp, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(spatial_shapes):
      # bs, H_*W_, num_heads, embed_dims ->
      # bs, H_*W_, num_heads*embed_dims ->
      # bs, num_heads*embed_dims, H_*W_ ->
      # bs*num_heads, embed_dims, H_, W_
      value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * self.num_heads, self.head_dims, int(H_.item()), int(W_.item()))
      # bs, num_queries, num_heads, num_points, 2 ->
      # bs, num_heads, num_queries, num_points, 2 ->
      # bs*num_heads, num_queries, num_points, 2
      sampling_grid_l_ = sampling_grids[:, :, :,level].transpose(1, 2).flatten(0, 1)
      # bs*num_heads, embed_dims, num_queries, num_points
      sampling_value_l_ = F.grid_sample(
        value_l_,
        sampling_grid_l_,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False)
      sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(bs * self.num_heads, 1, num_query, self.num_levels * self.num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(bs, self.num_heads * self.head_dims, num_query)
    return output.transpose(1, 2).contiguous()

class TemporalSelfAttention(nn.Module):
  """
    Args:
      embed_dims (int): The embedding dimension of attention. The same as inputs embedding dimension.
        Default: 256
      num_heads (int):  The number of heads.
        Default: 8
      num_levels (int): The number of scale levels
        Default: 4
      num_points (int): The number of sampling points
        Default: 4
      num_bev_queue (int): The number of BEVs to be used
        Default: 2 current BEV and last BEV 
      dropout (float): The drop out rate
        Default: 0.1
  """
  def __init__(self,embed_dims=256,num_heads=8,num_levels=4,num_points=4,num_bev_queue=2,dropout=0.1):
    super().__init__()
    self.embed_dims     = embed_dims
    self.num_heads      = num_heads
    self.num_levels     = num_levels
    self.num_points     = num_points
    self.num_bev_queue  = num_bev_queue
    self.xy             = 2
    self.dropout        = nn.Dropout(dropout)
    assert self.embed_dims % self.num_heads == 0, "embedding dimension must be divisible by number of heads"
    assert self.num_bev_queue>0, "value length must be larger than zero"
    self.head_dims  = self.embed_dims // self.num_heads
    self.to_Q = nn.Linear(self.embed_dims,self.embed_dims)
    self.to_K = nn.Linear(self.embed_dims,self.embed_dims)
    self.to_V = nn.Linear(self.embed_dims,self.embed_dims)
    self.sampling_offsets = nn.Linear(self.embed_dims, self.num_heads * self.num_bev_queue * self.num_points * self.num_levels * 2)
    self.output_proj = nn.Linear(self.embed_dims, self.embed_dims)
  def forward(self,query,key_hist=[],value_hist=[],reference_points=None,spatial_shapes=None):
    """
    Args:
      query (Tensor): Query of transformer with shape (bs, num_query, embed_dims)
      key_hist (list of Tensor): Key history of transformer with shape (bs, num_key, embed_dims). num_key should equel to num_levels * num_points
      value_hist (list of Tensor): Value history of transformer with shape (bs, num_value, embed_dims)
      reference_points (Tensor):  The normalized reference points with shape (bs, num_query, num_levels, 2),
        all elements are ranged in [0, 1], top-left (0,0),
        bottom-right (1, 1), including padding area.
        or (N, Length_{query}, num_levels, 4), add
        additional two dimensions is (w, h) to
        form reference boxes.
      spatial_shapes (Tensor): Spatial shape of features in different levels. With shape (num_levels, 2), last dimension represents (h, w).
    Returns:
      tensor with shape (bs, num_query, embed_dims)
    """
    query_list = [query for _ in range(self.num_bev_queue)] 
    key_list = key_hist
    value_list = value_hist
    key_list.insert(0,query)
    value_list.insert(0,query)
    assert len(query_list) == len(key_list) and len(key_list) == len(value_list) and len(value_list) == self.num_bev_queue, "length of query_list, key_list and value_list must equal to num_bev_queue"
    queries = torch.cat(query_list,dim=1)
    keies = torch.cat(key_list,dim=1)
    values = torch.cat(value_list,dim=1)
    bs, num_queryXnum_bev_queue, _ = queries.shape
    bs, num_keyXnum_bev_queue,   _ = keies.shape
    bs, num_valueXnum_bev_queue, _ = values.shape
    assert num_keyXnum_bev_queue   == self.num_levels * self.num_points * self.num_bev_queue,   "total key number does not match numbers of levels, points and bev queue"
    assert num_valueXnum_bev_queue == self.num_levels * self.num_points * self.num_bev_queue,   "total value number does not match numbers of levels, points and bev queue"
    #assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
    sampling_offsets = self.sampling_offsets(queries).view(bs, num_queryXnum_bev_queue, self.num_heads, self.num_points * self.num_bev_queue, self.num_levels, self.xy)
    q = self.to_Q(queries)  
    k = self.to_K(keies)    
    v = self.to_V(values)
    q = rearrange(queries, 'b q_bev (h d) -> b q_bev h d', h=self.num_heads)#q shape [bs, num_queryXnum_bev_queue, num_heads, head_dims] 
    k = rearrange(keies,   'b k_bev (h d) -> b k_bev h d', h=self.num_heads)#k shape [bs, num_keyXnum_bev_queue,   num_heads, head_dims]
    v = rearrange(values,  'b v_bev (h d) -> b v_bev h d', h=self.num_heads)#v shape [bs, num_valueXnum_bev_queue, num_heads, head_dims]
    print("kk shape ",k.shape)
    attention_weights = einsum(q, k, 'b q_bev h d, b k_bev h d -> b q_bev h k_bev')
    attention_weights = attention_weights.softmax(-1)
    attention_weights = rearrange(attention_weights, 'b q_bev h (l p_bev) -> b q_bev h l p_bev', l=self.num_levels)
    assert reference_points.shape[1]  == num_queryXnum_bev_queue // self.num_bev_queue, "second dim of reference_points must equal to num_query"
    reference_points_list = [reference_points for _ in range(self.num_bev_queue)] 
    reference_points = torch.cat(reference_points_list,dim=1)
    assert reference_points.shape[-1] == 2, "last dim of reference_points must be 2"
    """
    For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
    After proejcting, each BEV query has `num_Z_anchors == num_levels` reference points in each 2D image.
    For each referent point, we sample `num_points` sampling points.
    For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
    """
    offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
    sampling_locations = reference_points[:, :, None, None, :, :] + sampling_offsets / offset_normalizer[None, None, None, None, :, :]
    sampling_locations = sampling_locations.view(bs, num_queryXnum_bev_queue, self.num_heads, self.num_levels, self.num_points * self.num_bev_queue, self.xy)
    """
    multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights)
    """
    assert tuple(v.shape) == (bs, num_valueXnum_bev_queue, self.num_heads, self.head_dims)
    assert tuple(spatial_shapes.shape) == (self.num_levels, self.xy)
    assert tuple(sampling_locations.shape) == (bs, num_queryXnum_bev_queue, self.num_heads, self.num_levels, self.num_points * self.num_bev_queue, self.xy)
    assert tuple(attention_weights.shape) == (bs, num_queryXnum_bev_queue, self.num_heads, self.num_levels, self.num_points * self.num_bev_queue)
    tmp = [int(H_.item()) * int(W_.item()) for H_, W_ in spatial_shapes]
    value_list = v.split(tmp, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(spatial_shapes):
      # bs, H_*W_, num_heads, embed_dims ->
      # bs, H_*W_, num_heads*embed_dims ->
      # bs, num_heads*embed_dims, H_*W_ ->
      # bs*num_heads, embed_dims, H_, W_
      value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * self.num_heads, self.head_dims, int(H_.item()), int(W_.item()))
      # bs, num_queryXnum_bev_queue, num_heads, num_points * num_bev_queue, 2 ->
      # bs, num_heads, num_queryXnum_bev_queue, num_points * num_bev_queue, 2 ->
      # bs*num_heads, num_queryXnum_bev_queue,  num_points * num_bev_queue, 2
      sampling_grid_l_ = sampling_grids[:, :, :,level].transpose(1, 2).flatten(0, 1)
      # bs*num_heads, embed_dims, num_queryXnum_bev_queue, num_points * num_bev_queue
      sampling_value_l_ = F.grid_sample(
        value_l_,
        sampling_grid_l_,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False)
      sampling_value_list.append(sampling_value_l_)
    # (bs, num_queryXnum_bev_queue, num_heads, num_levels, num_points * num_bev_queue) ->
    # (bs, num_heads, num_queryXnum_bev_queue, num_levels, num_points * num_bev_queue) ->
    # (bs, num_heads, 1, num_queryXnum_bev_queue, num_levels*num_points*num_bev_queue)
    attention_weights = attention_weights.transpose(1, 2).reshape(bs * self.num_heads, 1, num_queryXnum_bev_queue, self.num_levels * self.num_points * self.num_bev_queue)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(bs, self.num_heads * self.head_dims, num_queryXnum_bev_queue//self.num_bev_queue, self.num_bev_queue).mean(-1).transpose(1, 2).contiguous() #output shape [bs, num_query, embed_dims]
    return self.dropout(output) + query

bs = 32
num_c = 6
num_q = 16
num_k = 16
num_v = 16
num_l = 4
num_p = 4
embed_dims = 256
num_h = 8
q = torch.rand(size=(bs,num_q,embed_dims))
k = torch.rand(size=(num_c,bs,num_k,embed_dims))
v = torch.rand(size=(num_c,bs,num_v,embed_dims))
reference_points = torch.rand(size=(bs,num_q,num_l,2))
spatial_shapes = torch.Tensor([[1,1],[2,2],[3,3],[1,2]])
reference_points_cam = torch.rand(num_c, bs, num_q, num_l, 2)
bev_mask = torch.rand(num_c, bs, num_q, num_l)
att = MSDeformableAttention3D()
sca = SpatialCrossAttention(deformable_attention=att)
res = sca(query=q,key=k,value=v,reference_points=reference_points,spatial_shapes=spatial_shapes,reference_points_cam=reference_points_cam,bev_mask=bev_mask)
print("sca ", res.shape)

q = torch.rand(size=(bs,num_q,embed_dims))
k = torch.rand(size=(bs,num_k,embed_dims))
v = torch.rand(size=(bs,num_v,embed_dims))
reference_points = torch.rand(size=(bs,num_q,num_l,2))
spatial_shapes = torch.Tensor([[1,2],[2,4],[3,6],[1,4]])
ta = TemporalSelfAttention()
res = ta(q,key_hist=[k],value_hist=[k],reference_points=reference_points,spatial_shapes=spatial_shapes)
print("ta ",res.shape)
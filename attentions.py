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
    self.NN_to_Q             = nn.Linear(self.head_dims,self.head_dims).to(device)
    self.NN_to_K             = nn.Linear(self.head_dims,self.head_dims).to(device)
    self.NN_to_V             = nn.Linear(self.head_dims,self.head_dims).to(device)
    self.NN_sampling_offsets = nn.Linear(self.head_dims,self.num_levels * self.num_points * self.num_zAnchors * 2).to(device)
    self.NN_output           = nn.Linear(self.embed_dims,self.embed_dims).to(device)
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
    with torch.no_grad():
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
      # ------------> [bs * num_cams, max_len, embed_dims]
      query_rebatch = query_rebatch.view(bs*self.num_cams, max_len, self.embed_dims)
      # reference_points_rebatch [bs, num_cams,  max_len, num_zAnchors, 2]
      # -----------------------> [bs * num_cams, max_len, num_zAnchors, 2]
      reference_points_rebatch = reference_points_rebatch.view(bs*self.num_cams, max_len, self.num_zAnchors, 2)
      # key, value [num_cams, bs,  num_key(or num_value), embed_dims]
      # ---------> [bs * num_cams, num_key(or num_value), embed_dims]
      key   = key.view(bs*self.num_cams,num_key,self.embed_dims)
      value = value.view(bs*self.num_cams,num_value,self.embed_dims)
      #####################################################################################################################################################################
      # start of multi-scale deformable attention computation #############################################################################################################
      #####################################################################################################################################################################
      bc, nq, _ = query_rebatch.shape
      bc, nk, _ = key.shape
      bc, nv, _ = value.shape
      # query_rebatch, key, value  [bs * num_cams, nq(or nk or nv), embed_dims]
      # ------------------------>  [bs * num_cams, nq(or nk or nv), num_heads, head_dims]
      query_rebatch = rearrange(query_rebatch, 'b q (h d) -> b q h d', h=self.num_heads) 
      key           = rearrange(key, 'b k (h d) -> b k h d', h=self.num_heads)
      value         = rearrange(value, 'b v (h d) -> b v h d', h=self.num_heads)
      # offset_normalizer [num_levels, 2]
      offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
      # reference_points_rebatch [bs * num_cams, nq, 1(to extend to num_heads), 1(to extend to num_levels), 1(to extend to num_points), num_zAnchor, 2]
      reference_points_rebatch = reference_points_rebatch[:, :, None, None, None, :, :]
      # query_rebatch      [bs * num_cams, nq(or nk or nv), num_heads, head_dims]
      # ->sampling_offsets [bs * num_cams, nq, num_heads, num_levels * num_points * num_zAnchors * 2]
      # -----------------> [bs * num_cams, nq, num_heads, num_levels,  num_points * num_zAnchors,  2]
      # -----------------> [bs * num_cams, nq, num_heads, num_levels,  num_points,  num_zAnchors,  2]
    sampling_offsets = self.NN_sampling_offsets(query_rebatch).view(bc, nq, self.num_heads, self.num_levels, self.num_points * self.num_zAnchors, self.xy)
    with torch.no_grad():
      sampling_offsets = torch.div(sampling_offsets, offset_normalizer[None, None, None, :, None, :])
      sampling_offsets = sampling_offsets.view(bs * num_cams, nq, self.num_heads, self.num_levels, self.num_points, self.num_zAnchors, self.xy)
      # sampling_locations [bs * num_cams, nq, num_heads, num_levels, num_points,  num_zAnchors, 2]
      # -----------------> [bs * num_cams, nq, num_heads, num_levels, num_points * num_zAnchors, 2]
      sampling_locations = torch.add(reference_points_rebatch, sampling_offsets)
      sampling_locations = sampling_locations.view(bs * num_cams, nq, self.num_heads, self.num_levels, self.num_points * self.num_zAnchors, self.xy)
      tmp        = [int(H_.item()) * int(W_.item()) for H_, W_ in spatial_shapes]
      key_list   = key.split(tmp, dim=1)
      value_list = value.split(tmp, dim=1)
      # sampling_grids [bs * num_cams, nq, num_heads, num_levels, num_points * num_zAnchors, 2]
      sampling_grids = torch.sub(torch.mul(2, sampling_locations), 1)
      sampling_key_list   = []
      sampling_value_list = []
      for level, (H_, W_) in enumerate(spatial_shapes):
        # key_l_ and value_l_ [bs * num_cams,             H_ * W_,                num_heads,              head_dims] 
        # ------------------> [bs * num_cams,             H_ * W_,                num_heads * head_dims]
        # ------------------> [bs * num_cams,             num_heads * head_dims,  H_ * W_]
        # ------------------> [bs * num_cams * num_heads, head_dims,              H_,                     W_]
        key_l_   =   key_list[level].flatten(2).transpose(1, 2).reshape(bs * self.num_cams * self.num_heads, self.head_dims, int(H_.item()), int(W_.item()))
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * self.num_cams * self.num_heads, self.head_dims, int(H_.item()), int(W_.item()))
        # sampling_grid_key_l_  [bs * num_cams,              1,         num_heads,                   num_points * num_zAnchors, 2]
        # ------------------->  [bs * num_cams,              num_heads,  1,                          num_points * num_zAnchors, 2]
        # ------------------->  [bs * num_cams * num_heads,  1,         num_points * num_zAnchors,   2]
        sampling_grid_key_l_   = sampling_grids[:, :1, :,level].transpose(1, 2).flatten(0, 1).contiguous()
        # sampling_key_l_   [bs * num_cams * num_heads,  head_dims,  1,  num_points * num_zAnchors]
        sampling_key_l_ = F.grid_sample(key_l_,sampling_grid_key_l_,mode='bilinear',padding_mode='zeros',align_corners=False)
        sampling_key_list.append(sampling_key_l_)
        # sampling_grid_value_l_  [bs * num_cams,              nq,         num_heads,                   num_points * num_zAnchors, 2]
        # --------------------->  [bs * num_cams,              num_heads,  nq,                          num_points * num_zAnchors, 2]
        # --------------------->  [bs * num_cams * num_heads,  nq,         num_points * num_zAnchors,   2]
        sampling_grid_value_l_ = sampling_grids[:, :, :,level].transpose(1, 2).flatten(0, 1).contiguous()
        # sampling_value_l_ [bs * num_cams * num_heads,  head_dims,  nq,  num_points * num_zAnchors]
        sampling_value_l_ = F.grid_sample(value_l_,sampling_grid_value_l_,mode='bilinear',padding_mode='zeros',align_corners=False)
        sampling_value_list.append(sampling_value_l_)
      # sampling_key_list  [[bs * num_cams * num_heads, head_dims, 1, num_points * num_zAnchors]....]
      # ---------------->   [bs * num_cams * num_heads, head_dims, 1, num_levels(get the dimension by stack in dim -2), num_points * num_zAnchors]
      # ---------------->   [bs * num_cams * num_heads, head_dims, 1 * num_levels * num_points * num_zAnchors(by flatten)]
      # ---------------->   [bs * num_cams,  num_heads, head_dims, 1 * num_levels * num_points * num_zAnchors]
      # ---------------->   [bs * num_cams,  1 * num_levels * num_points * num_zAnchors, num_heads, head_dims]
      key_sampled   = torch.stack(sampling_key_list, dim=-2).flatten(-3).view(bc, self.num_heads, self.head_dims, 1 * self.num_levels * self.num_points * self.num_zAnchors).permute(0,3,1,2).contiguous()
      # sampling_value_list  [[bs * num_cams * num_heads, head_dims, nq, num_points * num_zAnchors]....]
      # ------------------>   [bs * num_cams * num_heads, head_dims, nq, num_levels(get the dimension by stack in dim -2), num_points * num_zAnchors]
      # ------------------>   [bs * num_cams * num_heads, head_dims, nq, num_levels * num_points * num_zAnchors(by flatten)]
      value_sampled = torch.stack(sampling_value_list, dim=-2).flatten(-2)
      assert key_sampled.shape[1]   == 1 * self.num_levels * self.num_points * self.num_zAnchors,   "number of sampled key corresponding to one query   must match numbers of levels * points * zAnchors"
      assert value_sampled.shape[-1] == self.num_levels * self.num_points * self.num_zAnchors,      "number of sampled value corresponding to one query must match numbers of levels * points * zAnchors"
      assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == nv
    # q, k [bs * num_cams, nq(or num_levels * num_points * num_zAnchors), num_heads, head_dims]
    q = self.NN_to_Q(query_rebatch) 
    k = self.NN_to_K(key_sampled)    
    # v  [bs * num_cams * num_heads, head_dims, nq, num_levels * num_points * num_zAnchors]
    # -> [bs * num_cams * num_heads, nq, num_levels * num_points * num_zAnchors, head_dims]
    # -> [bs * num_cams * num_heads, head_dims, nq, num_levels * num_points * num_zAnchors]
    v = self.NN_to_V(value_sampled.permute(0,2,3,1)).permute(0,3,1,2).contiguous()
    # attention_weights [bs * num_cams, nq, num_heads, num_levels * num_points * num_zAnchors]
    # ----------------> [bs * num_cams, nq, num_heads, num_levels,  num_points * num_zAnchors]
    # ----------------> [bs * num_cams, num_heads,     nq,          num_levels,  num_points * num_zAnchors]
    # ----------------> [bs * num_cams * num_heads,    1(to extend to head_dims), nq, num_levels * num_points * num_zAnchors]
    with torch.no_grad():
      attention_weights = einsum(q, k, 'b q h d, b k h d -> b q h k')
      attention_weights = attention_weights.softmax(-1)
      attention_weights = rearrange(attention_weights, 'b q h (l pz) -> b q h l pz', l=self.num_levels, pz=self.num_points * self.num_zAnchors)
      attention_weights = attention_weights.transpose(1, 2).reshape(bc * self.num_heads, 1, nq, self.num_levels * self.num_points * self.num_zAnchors)
      assert reference_points_rebatch.shape[1]  == nq, "second dim of reference_points must equal to nq"
      assert reference_points_rebatch.shape[-1] == 2,  "last dim of reference_points must be 2"    
      """
      multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights)
      """
      assert tuple(v.shape) == (bc * self.num_heads, self.head_dims, nq, self.num_levels * self.num_points * self.num_zAnchors)
      assert tuple(spatial_shapes.shape) == (self.num_levels, self.xy)
      assert tuple(sampling_locations.shape) == (bc, nq, self.num_heads, self.num_levels, self.num_points * self.num_zAnchors, self.xy)
      assert tuple(attention_weights.shape)  == (bc * self.num_heads, 1, nq, self.num_levels * self.num_points * self.num_zAnchors)
      # attention [bs * num_cams * num_heads, head_dims, nq, 1(by sum after elementwise multiplication)]
      # --------> [bs * num_cams, num_heads * head_dims(==embed_dims), nq]
      # --------> [bs * num_cams, nq, embed_dims]
      attention = (v * attention_weights).sum(-1).view(bc, self.num_heads * self.head_dims, nq).transpose(1, 2).contiguous()
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
          slots[j, index_query_per_img] = torch.add(slots[j, index_query_per_img], attention[j, i, :len(index_query_per_img)])
      # bev_mask [num_cam, bs, num_query, num_levels]
      # -->count [num_cam, bs, num_query]
      # -------> [bs, num_query]
      count = bev_mask.sum(-1) > 0
      count = count.permute(1, 2, 0).sum(-1)
      count = torch.clamp(count, min=1.0)
      # slots [bs, num_query, embed_dims]
      # count[..., None] makes count [bs, num_query, 1(to extend to embed_dims)]
      slots = torch.div(slots , count[..., None])
    slots = self.NN_output(slots)
    # return spatial attention [bs, num_query, embed_dims]
    # del count, attention, v, attention_weights, sampling_locations, reference_points_rebatch, \
    #     q, k, key_sampled, value_sampled, query_rebatch, \
    #     sampling_value_list, sampling_value_l_, sampling_grid_value_l_, value_l_, \
    #     sampling_key_list, sampling_key_l_, sampling_grid_key_l_, key_l_, \
    #     sampling_grids, key_list, value_list
        
    return torch.add(self.NN_dropout(slots), residual)



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
      Default: 1   
    num_points    (int):   The number of sampling points in a single level
      Default: 4
    -----Device-----
    device (torch.device): The device
      Default: cpu
  -------------------------------------------------------------------------------------------------
  """
  def __init__(self,num_sequences=2,dropout=0.1,embed_dims=256,num_heads=8,num_levels=1,num_points=4,device=torch.device("cpu")):
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
    self.NN_to_Q             = nn.Linear(self.head_dims,self.head_dims).to(device)
    self.NN_to_K             = nn.Linear(self.head_dims,self.head_dims).to(device)
    self.NN_to_V             = nn.Linear(self.head_dims,self.head_dims).to(device)
    self.NN_sampling_offsets = nn.Linear(self.head_dims,self.num_levels * self.num_points * 2).to(device)
    self.NN_output           = nn.Linear(self.embed_dims,self.embed_dims).to(device)
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
    with torch.no_grad():
      residual   = query
      query_list = [query for _ in range(self.num_sequences)] 
      key_list   = key_hist
      value_list = value_hist
      #key_list.insert(0,query)
      #value_list.insert(0,query)
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
      # reference_points [bs, num_query,                 num_levels, 2]
      # ---------------> [bs * num_sequences, num_query, num_levels, 2]
      reference_points_list = [reference_points for _ in range(self.num_sequences)] 
      reference_points = torch.cat(reference_points_list,dim=0).view(bs * self.num_sequences, num_query, self.num_levels,2)
      #####################################################################################################################################################################
      # start of attention computation ####################################################################################################################################
      #####################################################################################################################################################################
      # queries, keies, values  [bs * num_sequences, num_query(or num_key or num_value), embed_dims]
      # --------------------->  [bs * num_sequences, num_query(or num_key or num_value), num_heads, head_dims]
      queries = rearrange(queries, 'b q (h d) -> b q h d', h=self.num_heads) 
      keies   = rearrange(keies, 'b k (h d) -> b k h d', h=self.num_heads)
      values  = rearrange(values, 'b v (h d) -> b v h d', h=self.num_heads)
      # offset_normalizer [num_levels, 2]
      offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
      # sampling_offsets [bs * num_sequences, num_query, num_heads, num_levels, num_points, 2]
    sampling_offsets = self.NN_sampling_offsets(queries).view(bs * self.num_sequences, num_query, self.num_heads, self.num_levels, self.num_points, self.xy)
    with torch.no_grad():
      assert reference_points.shape[1]  == num_query, "second dim of reference_points must equal to num_query"
      assert reference_points.shape[-1] == 2,         "last dim of reference_points must be 2"
      # reference_points  extends to [bs * num_sequences, num_query, 1(to extend to num_heads), num_levels, 1(to extend to num_points), 2]
      # offset_normalizer extends to [bs * num_sequences, num_query, num_heads, num_levels, num_points, 2]
      sampling_locations = torch.add(reference_points[:, :, None, :, None, :], torch.div(sampling_offsets, offset_normalizer[None, None, None, :, None, :]))
      tmp        = [int(H_.item()) * int(W_.item()) for H_, W_ in spatial_shapes]
      key_list   = keies.split(tmp, dim=1)
      value_list = values.split(tmp, dim=1)
      # sampling_grids [bs * num_sequences, num_query, num_heads, num_levels, num_points, 2]
      sampling_grids = torch.sub(torch.mul(2, sampling_locations), 1)
      sampling_key_list   = []
      sampling_value_list = []
      for level, (H_, W_) in enumerate(spatial_shapes):
        # key_l_ and value_l_ [bs * num_sequences,             H_ * W_,                num_heads,              head_dims] 
        # ------------------> [bs * num_sequences,             H_ * W_,                num_heads * head_dims]
        # ------------------> [bs * num_sequences,             num_heads * head_dims,  H_ * W_]
        # ------------------> [bs * num_sequences * num_heads, head_dims,              H_,                     W_]
        key_l_   =   key_list[level].flatten(2).transpose(1, 2).reshape(bs * self.num_sequences * self.num_heads, self.head_dims, int(H_.item()), int(W_.item()))
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * self.num_sequences * self.num_heads, self.head_dims, int(H_.item()), int(W_.item()))
        # sampling_grid_key_l_  [bs * num_sequences,              1,         num_heads,                   num_points, 2]
        # ------------------->  [bs * num_sequences,              num_heads,  1,                          num_points, 2]
        # ------------------->  [bs * num_sequences * num_heads,  1,         num_points,   2]
        sampling_grid_key_l_   = sampling_grids[:, :1, :,level].transpose(1, 2).flatten(0, 1).contiguous()
        # sampling_key_l_   [bs * num_sequences * num_heads,  head_dims,  1,  num_points]
        sampling_key_l_ = F.grid_sample(key_l_,sampling_grid_key_l_,mode='bilinear',padding_mode='zeros',align_corners=False)
        sampling_key_list.append(sampling_key_l_)
        # sampling_grid_value_l_  [bs * num_sequences,              n_query,         num_heads,                   num_points, 2]
        # --------------------->  [bs * num_sequences,              num_heads,       n_query,                     num_points, 2]
        # --------------------->  [bs * num_sequences * num_heads,  n_query,         num_points ,   2]
        sampling_grid_value_l_ = sampling_grids[:, :, :,level].transpose(1, 2).flatten(0, 1).contiguous()
        # sampling_value_l_ [bs * num_sequences * num_heads,  head_dims,  n_query,  num_points ]
        sampling_value_l_ = F.grid_sample(value_l_,sampling_grid_value_l_,mode='bilinear',padding_mode='zeros',align_corners=False)
        sampling_value_list.append(sampling_value_l_)
      # sampling_key_list  [[bs * num_sequences * num_heads, head_dims, 1, num_points ]....]
      # ---------------->   [bs * num_sequences * num_heads, head_dims, 1, num_levels(get the dimension by stack in dim -2), num_points]
      # ---------------->   [bs * num_sequences * num_heads, head_dims, 1 * num_levels * num_points (by flatten)]
      # ---------------->   [bs * num_sequences,  num_heads, head_dims, 1 * num_levels * num_points]
      # ---------------->   [bs * num_sequences,  1 * num_levels * num_points, num_heads, head_dims]
      key_sampled   = torch.stack(sampling_key_list, dim=-2).flatten(-3).view(bs * self.num_sequences, self.num_heads, self.head_dims, 1 * self.num_levels * self.num_points).permute(0,3,1,2).contiguous()
      # sampling_value_list  [[bs * num_sequences * num_heads, head_dims, n_query, num_points]....]
      # ------------------>   [bs * num_sequences * num_heads, head_dims, n_query, num_levels(get the dimension by stack in dim -2), num_points]
      # ------------------>   [bs * num_sequences * num_heads, head_dims, n_query, num_levels * num_points(by flatten)]
      value_sampled = torch.stack(sampling_value_list, dim=-2).flatten(-2)
      assert key_sampled.shape[1]    == 1 * self.num_levels * self.num_points,  "number of sampled key corresponding to one query   must match numbers of levels * points * zAnchors"
      assert value_sampled.shape[-1] == self.num_levels * self.num_points,      "number of sampled value corresponding to one query must match numbers of levels * points * zAnchors"
      assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
    # q, k [bs * num_sequences, num_query(or num_levels * num_points), num_heads, head_dims]
    q = self.NN_to_Q(queries) 
    k = self.NN_to_K(key_sampled)    
    # v  [bs * num_sequences * num_heads, head_dims, num_query, num_levels * num_points]
    # -> [bs * num_sequences * num_heads, num_query, num_levels * num_points, head_dims]
    # -> [bs * num_sequences * num_heads, head_dims, num_query, num_levels * num_points]
    v = self.NN_to_V(value_sampled.permute(0,2,3,1)).permute(0,3,1,2).contiguous()
    # attention_weights [bs * num_sequences, num_query, num_heads, num_levels * num_points]
    # ----------------> [bs * num_sequences, num_query, num_heads, num_levels,  num_points]
     # ---------------->[bs * num_sequences, num_heads,     num_query,          num_levels,  num_points]
    # ----------------> [bs * num_sequences * num_heads,    1(to extend to head_dims), num_query, num_levels * num_points]
    with torch.no_grad():
      attention_weights = einsum(q, k, 'b q h d, b k h d -> b q h k')
      attention_weights = attention_weights.softmax(-1)
      attention_weights = rearrange(attention_weights, 'b q h (l p) -> b q h l p', l=self.num_levels)
      attention_weights = attention_weights.transpose(1, 2).reshape(bs * self.num_sequences * self.num_heads, 1, num_query, self.num_levels * self.num_points)
      """
      multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights)
      """
      assert tuple(v.shape)                  == (bs * self.num_sequences * self.num_heads, self.head_dims, num_query, self.num_levels * self.num_points)
      assert tuple(spatial_shapes.shape)     == (self.num_levels, self.xy)
      assert tuple(sampling_locations.shape) == (bs * self.num_sequences, num_query, self.num_heads, self.num_levels, self.num_points, self.xy)
      assert tuple(attention_weights.shape)  == (bs * self.num_sequences * self.num_heads, 1, num_query, self.num_levels * self.num_points)
      # attention [bs * num_sequences * num_heads, head_dims, nq, 1(by sum after elementwise multiplication)]
      # --------> [bs * num_sequences, num_heads * head_dims(==embed_dims), nq]
      # --------> [bs * num_sequences, nq, embed_dims]
      attention = torch.mul(v, attention_weights).sum(-1).view(bs * self.num_sequences, self.num_heads * self.head_dims, num_query).transpose(1, 2).contiguous()
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
    return torch.add(self.NN_dropout(self.NN_output(attention)), residual)


class CustomAttention(nn.Module):
  """
  Args:
    dropout       (float): The drop out rate
      Default: 0.1
    embed_dims    (int):   The embedding dimension of attention. The same as inputs embedding dimension.
      Default: 256   
    num_heads     (int):   The number of heads.
      Default: 8   
    num_levels    (int):   The number of scale levels in a single sequence
      Default: 1   
    num_points    (int):   The number of sampling points in a single level
      Default: 4
    -----Device-----
    device (torch.device): The device
      Default: cpu
  -------------------------------------------------------------------------------------------------
  """
  def __init__(self,dropout=0.1,embed_dims=256,num_heads=8,num_levels=1,num_points=4,device=torch.device("cpu")):
    super().__init__()
    self.dropout       = dropout
    self.embed_dims    = embed_dims
    self.num_heads     = num_heads
    self.num_levels    = num_levels
    self.num_points    = num_points
    self.device        = device
    self.xy            = 2
    assert self.embed_dims % self.num_heads == 0, "embedding dimension must be divisible by number of heads"
    self.head_dims      = self.embed_dims // self.num_heads
    self.NN_to_Q             = nn.Linear(self.head_dims,self.head_dims).to(device)
    self.NN_to_K             = nn.Linear(self.head_dims,self.head_dims).to(device)
    self.NN_to_V             = nn.Linear(self.head_dims,self.head_dims).to(device)
    self.NN_sampling_offsets = nn.Linear(self.head_dims,self.num_levels * self.num_points * 2).to(device)
    self.NN_output           = nn.Linear(self.embed_dims,self.embed_dims).to(device)
    self.NN_dropout          = nn.Dropout(dropout).to(device)
  def forward(self,query,key,value,reference_points=None,spatial_shapes=None):
    """
    Args:
      query             (Tensor         [bs, num_query, embed_dims]):    The query
      key               (Tensor         [bs, num_key,   embed_dims]):    The query
      value             (Tensor         [bs, num_value, embed_dims]):    The query
      reference_points  (Tensor         [bs, num_query, num_levels, 2]): The normalized reference points. Passed though to multi scale deformable attention layer
      spatial_shapes    (Tensor         [num_levels, 2]):                The Spatial shape of features in different levels. Passed though to multi scale deformable attention layer
    Returns:
      forwarded result  (tensor         [bs, num_query, embed_dims])
    """
    with torch.no_grad():
      # residual [bs, num_query, embed_dims]
      residual   = query
      #####################################################################################################################################################################
      # start of attention computation ####################################################################################################################################
      #####################################################################################################################################################################
      bs, num_query, _ = query.shape
      bs, num_key,   _ = value.shape
      bs, num_value, _ = value.shape
      # query, key, value  [bs, num_query(or num_key or num_value), embed_dims]
      # ---------------->  [bs, num_query(or num_key or num_value), num_heads, head_dims]
      query = rearrange(query, 'b q (h d) -> b q h d', h=self.num_heads) 
      key   = rearrange(key,   'b k (h d) -> b k h d', h=self.num_heads)
      value = rearrange(value, 'b v (h d) -> b v h d', h=self.num_heads)
      # offset_normalizer [num_levels, 2]
      offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
      # sampling_offsets [bs, num_query, num_heads, num_levels, num_points, 2]
    sampling_offsets = self.NN_sampling_offsets(query).view(bs, num_query, self.num_heads, self.num_levels, self.num_points, self.xy)
    with torch.no_grad():
      assert reference_points.shape[1]  == num_query, "second dim of reference_points must equal to num_query"
      assert reference_points.shape[-1] == 2,         "last dim of reference_points must be 2"
      # reference_points  extends to [bs, num_query, 1(to extend to num_heads), num_levels, 1(to extend to num_points), 2]
      # offset_normalizer extends to [bs, num_query, num_heads, num_levels, num_points, 2]
      sampling_locations = torch.add(reference_points[:, :, None, :, None, :], torch.div(sampling_offsets, offset_normalizer[None, None, None, :, None, :]))
      tmp        = [int(H_.item()) * int(W_.item()) for H_, W_ in spatial_shapes]
      key_list   = key.split(tmp, dim=1)
      value_list = value.split(tmp, dim=1)
      # sampling_grids [bs, num_query, num_heads, num_levels, num_points, 2]
      sampling_grids = torch.sub(torch.mul(2, sampling_locations), 1)
      sampling_key_list   = []
      sampling_value_list = []
      for level, (H_, W_) in enumerate(spatial_shapes):
        # key_l_ and value_l_ [bs,             H_ * W_,                num_heads,              head_dims] 
        # ------------------> [bs,             H_ * W_,                num_heads * head_dims]
        # ------------------> [bs,             num_heads * head_dims,  H_ * W_]
        # ------------------> [bs * num_heads, head_dims,              H_,                     W_]
        key_l_   =   key_list[level].flatten(2).transpose(1, 2).reshape(bs * self.num_heads, self.head_dims, int(H_.item()), int(W_.item()))
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * self.num_heads, self.head_dims, int(H_.item()), int(W_.item()))
        # sampling_grid_key_l_  [bs,              1,         num_heads,                   num_points, 2]
        # ------------------->  [bs,              num_heads,  1,                          num_points, 2]
        # ------------------->  [bs * num_heads,  1,         num_points,   2]
        sampling_grid_key_l_   = sampling_grids[:, :1, :,level].transpose(1, 2).flatten(0, 1).contiguous()
        # sampling_key_l_   [bs * num_heads,  head_dims,  1,  num_points]
        sampling_key_l_ = F.grid_sample(key_l_,sampling_grid_key_l_,mode='bilinear',padding_mode='zeros',align_corners=False)
        sampling_key_list.append(sampling_key_l_)
        # sampling_grid_value_l_  [bs,              n_query,         num_heads,                   num_points, 2]
        # --------------------->  [bs,              num_heads,       n_query,                     num_points, 2]
        # --------------------->  [bs * num_heads,  n_query,         num_points ,   2]
        sampling_grid_value_l_ = sampling_grids[:, :, :,level].transpose(1, 2).flatten(0, 1).contiguous()
        # sampling_value_l_ [bs * num_heads,  head_dims,  n_query,  num_points ]
        sampling_value_l_ = F.grid_sample(value_l_,sampling_grid_value_l_,mode='bilinear',padding_mode='zeros',align_corners=False)
        sampling_value_list.append(sampling_value_l_)
      # sampling_key_list  [[bs * num_heads, head_dims, 1, num_points ]....]
      # ----->key_sampled   [bs * num_heads, head_dims, 1, num_levels(get the dimension by stack in dim -2), num_points]
      # ---------------->   [bs * num_heads, head_dims, 1 * num_levels * num_points (by flatten)]
      # ---------------->   [bs,  num_heads, head_dims, 1 * num_levels * num_points]
      # ---------------->   [bs,  1 * num_levels * num_points, num_heads, head_dims]
      key_sampled   = torch.stack(sampling_key_list, dim=-2).flatten(-3).view(bs, self.num_heads, self.head_dims, 1 * self.num_levels * self.num_points).permute(0,3,1,2).contiguous()
      # sampling_value_list  [[bs * num_heads, head_dims, n_query, num_points]....]
      # ----->value_sampled   [bs * num_heads, head_dims, n_query, num_levels(get the dimension by stack in dim -2), num_points]
      # ------------------>   [bs * num_heads, head_dims, n_query, num_levels * num_points(by flatten)]
      value_sampled = torch.stack(sampling_value_list, dim=-2).flatten(-2)
      assert key_sampled.shape[1]    == 1 * self.num_levels * self.num_points,  "number of sampled key corresponding to one query   must match numbers of levels * points * zAnchors"
      assert value_sampled.shape[-1] == self.num_levels * self.num_points,      "number of sampled value corresponding to one query must match numbers of levels * points * zAnchors"
      assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
    # q, k [bs, num_query(or num_levels * num_points), num_heads, head_dims]
    q = self.NN_to_Q(query) 
    k = self.NN_to_K(key_sampled)    
    # v  [bs * num_heads, head_dims, num_query, num_levels * num_points]
    # -> [bs * num_heads, num_query, num_levels * num_points, head_dims]
    # -> [bs * num_heads, head_dims, num_query, num_levels * num_points]
    v = self.NN_to_V(value_sampled.permute(0,2,3,1).contiguous()).permute(0,3,1,2).contiguous()
    # attention_weights [bs, num_query, num_heads, num_levels * num_points]
    # ----------------> [bs, num_query, num_heads, num_levels,  num_points]
     # ---------------->[bs, num_heads,     num_query,          num_levels,  num_points]
    # ----------------> [bs * num_heads,    1(to extend to head_dims), num_query, num_levels * num_points]
    with torch.no_grad():
      attention_weights = einsum(q, k, 'b q h d, b k h d -> b q h k').detach()
      attention_weights = attention_weights.softmax(-1)
      attention_weights = rearrange(attention_weights, 'b q h (l p) -> b q h l p', l=self.num_levels)
      attention_weights = attention_weights.transpose(1, 2).reshape(bs * self.num_heads, 1, num_query, self.num_levels * self.num_points)
      """
      multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights)
      """
      assert tuple(v.shape)                  == (bs * self.num_heads, self.head_dims, num_query, self.num_levels * self.num_points)
      assert tuple(spatial_shapes.shape)     == (self.num_levels, self.xy)
      assert tuple(sampling_locations.shape) == (bs, num_query, self.num_heads, self.num_levels, self.num_points, self.xy)
      assert tuple(attention_weights.shape)  == (bs * self.num_heads, 1, num_query, self.num_levels * self.num_points)
      # attention [bs * num_heads, head_dims, nq, 1(by sum after elementwise multiplication)]
      # --------> [bs, num_heads * head_dims(==embed_dims), nq]
      # --------> [bs, nq, embed_dims]
      attention = torch.mul(v, attention_weights).sum(-1).view(bs, self.num_heads * self.head_dims, num_query).transpose(1, 2).contiguous()
    #####################################################################################################################################################################
    # end of attention computation ######################################################################################################################################
    #####################################################################################################################################################################
    # attention [bs, num_query, embed_dims]
    # return custom attention [bs, num_query, embed_dims]
    return torch.add(self.NN_dropout(self.NN_output(attention)), residual)


class FullAttention(nn.Module):
  """
  Args:
    dropout       (float): The drop out rate
      Default: 0.1
    embed_dims    (int):   The embedding dimension of attention. The same as inputs embedding dimension.
      Default: 256   
    num_heads     (int):   The number of heads.
      Default: 8   
    num_levels    (int):   The number of scale levels in a single sequence
      Default: 1   
    num_points    (int):   The number of sampling points in a single level
      Default: 4
    -----Device-----
    device (torch.device): The device
      Default: cpu
  -------------------------------------------------------------------------------------------------
  """
  def __init__(self,dropout=0.1,embed_dims=256,num_heads=8,num_levels=1,num_points=4,device=torch.device("cpu")):
    super().__init__()
    self.dropout       = dropout
    self.embed_dims    = embed_dims
    self.num_heads     = num_heads
    self.num_levels    = num_levels
    self.num_points    = num_points
    self.device        = device
    assert self.embed_dims % self.num_heads == 0, "embedding dimension must be divisible by number of heads"
    self.head_dims      = self.embed_dims // self.num_heads
    self.NN_to_Q             = nn.Linear(self.head_dims,self.head_dims).to(device)
    self.NN_to_K             = nn.Linear(self.head_dims,self.head_dims).to(device)
    self.NN_to_V             = nn.Linear(self.head_dims,self.head_dims).to(device)
    #self.NN_sampling_offsets = nn.Linear(self.head_dims,self.num_levels * self.num_points * 2).to(device)
    self.NN_output           = nn.Linear(self.embed_dims,self.embed_dims).to(device)
    self.NN_dropout          = nn.Dropout(dropout).to(device)
  def forward(self,query,key,value):
    """
    Args:
      query             (Tensor         [bs, num_query, embed_dims]):    The query
      key               (Tensor         [bs, num_key,   embed_dims]):    The query
      value             (Tensor         [bs, num_value, embed_dims]):    The query
    Returns:
      forwarded result  (tensor         [bs, num_query, embed_dims])
    """
    with torch.no_grad():
      # residual [bs, num_query, embed_dims]
      residual   = query
      bs, num_query, _ = query.shape
      bs, num_key,   _ = value.shape
      bs, num_value, _ = value.shape
      assert num_key == self.num_levels * self.num_points and num_value == self.num_levels * self.num_points, "in the leveled full attention, both num_key and num_value must equal to num_levels * num_points"
      # query, key, value  [bs, num_query(or num_key or num_value), embed_dims]
      # ---------------->  [bs, num_query(or num_key or num_value), num_heads, head_dims]
      query = rearrange(query, 'b q (h d) -> b q h d', h=self.num_heads) 
      key   = rearrange(key,   'b k (h d) -> b k h d', h=self.num_heads)
      value = rearrange(value, 'b v (h d) -> b v h d', h=self.num_heads)
    # q, k [bs, num_query(or num_key or num_value), num_heads, head_dims]
    q = self.NN_to_Q(query) 
    k = self.NN_to_K(key)    
    v = self.NN_to_V(value)
    # attention_weights [bs, num_query, num_heads, num_levels * num_points]
    # ----------------> [bs, num_query, num_heads, num_levels,  num_points]
     # ---------------->[bs, num_heads,     num_query,          num_levels,  num_points]
    # ----------------> [bs * num_heads,    1(to extend to head_dims), num_query, num_levels * num_points]
    with torch.no_grad():
      attention_weights = einsum(q, k, 'b q h d, b k h d -> b q h k')
      attention_weights = attention_weights.softmax(-1)
      attention = rearrange(einsum(attention_weights, v, 'b q h v, b v h d -> b q h d'),'b q h d -> b q (h d)')
    # return full attention [bs, num_query, embed_dims]
    return torch.add(self.NN_dropout(self.NN_output(attention)), residual)

class AddAndNormLayer(nn.Module):
  """
  Args:
    num_query (int): The number of query
      Default: None
    embed_dims(int): The embedding dimension
      Default: 6
    -----Device-----
    device (torch.device): The device
      Default: cpu
  -------------------------------------------------------------------------------------------------
  Note:
    1, Input of this layer is of shape [bs, num_query, embed_dims]
    2, This implementation can normalize last two dimensions, which are num_query and embed_dims, if num_query is provided.
       Otherwise, only the last dimension which is embed_dims is normalized
  """
  def __init__(self,num_query=None, embed_dims=256,device=torch.device("cpu")):
    super().__init__()
    self.num_query  = num_query
    self.embed_dims = embed_dims
    self.device     = device
    if num_query is not None:
      self.NN_layerNorm = nn.LayerNorm([num_query,embed_dims],device=device)
    else:
      self.NN_layerNorm = nn.LayerNorm(embed_dims,device=device)
  def forward(self,x,y):
    """
    Args:
      x       (Tensor [bs, num_query, emded_dims]): The tensor to be added
      y       (Tensor [bs, num_query, emded_dims]): The tensor to add
    Returns:
      addNorm (Tensor [bs, num_query, emded_dims])
    """
    added = torch.add(x, y)
    return self.NN_layerNorm(added)
# to just test dimension "assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value" in MultiScaleDeformableAttention3D needs to be commented out
# in normal use, the assert should ramain

from attentions import *

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
sca = SpatialCrossAttention()
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
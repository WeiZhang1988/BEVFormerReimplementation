# to just test dimension "assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value" in MultiScaleDeformableAttention3D needs to be commented out
# in normal use, the assert should ramain

from attentions import *
from vovnet import *
from encoder import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bs = 32
num_c = 6
num_q = 16
num_k = 16
num_v = 16
num_l = 4
num_p = 4
embed_dims = 256
num_h = 8
q_spat = torch.rand(size=(bs,num_q,embed_dims)).to(device)
k_spat = torch.rand(size=(num_c,bs,num_k,embed_dims)).to(device)
v_spat = torch.rand(size=(num_c,bs,num_v,embed_dims)).to(device)
reference_points = torch.rand(size=(bs,num_q,num_l,2)).to(device)
spatial_shapes = torch.Tensor([[1,1],[2,2],[3,3],[1,2]]).to(device)
reference_points_cam = torch.rand(num_c, bs, num_q, num_l, 2).to(device)
bev_mask = torch.rand(num_c, bs, num_q, num_l).to(device)
sca = SpatialCrossAttention(device=device)
res = sca(query=q_spat,key=k_spat,value=v_spat,reference_points=reference_points,spatial_shapes=spatial_shapes,reference_points_cam=reference_points_cam,bev_mask=bev_mask)
print("sca ", res.shape)
q_temp = torch.rand(size=(bs,num_q,embed_dims)).to(device)
k_temp = torch.rand(size=(bs,num_k,embed_dims)).to(device)
v_temp = torch.rand(size=(bs,num_v,embed_dims)).to(device)
reference_points = torch.rand(size=(bs,num_q,num_l,2)).to(device)
spatial_shapes = torch.Tensor([[1,1],[2,2],[3,3],[1,2]]).to(device)
ta = TemporalSelfAttention(device=device)
res = ta(q_temp,key_hist=[k_temp],value_hist=[k_temp],reference_points=reference_points,spatial_shapes=spatial_shapes)
print("ta ",res.shape)
reference_points = torch.rand(size=(bs,num_q,num_l,2)).to(device)
spatial_shapes = torch.Tensor([[1,1],[2,2],[3,3],[1,2]]).to(device)
bev = BEVFormerLayer(num_bev_queue=2,device=device)
res = bev(k_spat,v_spat,q_temp,tempAttn_key_hist=[k_temp],tempAttn_value_hist=[v_temp],reference_points=reference_points,spatial_shapes=spatial_shapes,reference_points_cam=reference_points_cam,bev_mask=bev_mask)
print("bev ",res.shape)


x = torch.rand(size=(32,3,96,96)).to(device)
vov = VoVNet(device=device)
res = vov(x)
print("vov ", res.shape)
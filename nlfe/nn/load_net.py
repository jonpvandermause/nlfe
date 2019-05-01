import torch
from nets import OneLayer, TwoLayer, ThreeLayer

net_ex = torch.load('saved_models/one_layer_0')
print(net_ex(torch.tensor([[3., 1.]])))

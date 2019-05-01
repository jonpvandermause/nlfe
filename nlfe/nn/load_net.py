import torch
from train_network import Net

net_ex = torch.load('net_test')
print(net_ex(torch.tensor([[3., 1.]])))
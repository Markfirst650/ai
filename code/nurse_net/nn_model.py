import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        y = x + 1
        return y

my_net = Net()
x = torch.tensor(1)
y = my_net(x)
print(y)

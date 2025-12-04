import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.nn.functional import relu, log_softmax


#构建网络框架
class Net (nn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = nn.Linear(28 * 28,64)#------图片是28*28像素，且是黑白图片，只有1层
        self.f2 = nn.Linear(64,64)
        self.f3 = nn.Linear(64,64)
        self.f4 = nn.Linear(64,10)#----输出10个类别

    def forward(self,x):
        x = relu(self.f1(x))
        x = relu(self.f2(x))
        x = relu(self.f3(x))
        x = log_softmax(self.f4(x), dim=1)
        return x



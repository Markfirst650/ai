import torch
from torch import nn #神经网络模型的框架需要
from torch.nn import Conv2d,MaxPool2d,Flatten, Linear #网络中的系列运算需要
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./data',train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=64,shuffle=True)
class solmount (nn.Module):
    def __init__(self):
        super(solmount, self).__init__()
        self.conv1 = Conv2d(3,32,5,padding=2)
        self.maxpool = nn.MaxPool2d(2)
        self.conv2 = Conv2d(32,32,5,padding=2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = Conv2d(32,64,5,padding=2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.flatten = Flatten()
        self.linear = Linear(1024,64)
        self.linear2 = Linear(64,10)
    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.linear2(x)
        return x

Solmount = solmount()
print(Solmount)
input = torch.ones(64,3,32,32)
output = Solmount(input)
print(output)
print(output.shape)

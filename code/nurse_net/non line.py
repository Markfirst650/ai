import torch
from torch import nn
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torchvision

dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

class Kong(nn.Module):
    def __init__(self):
        super(Kong,self).__init__()
        self.relu = nn.ReLU(True)

    def forward(self,x):
        x = self.relu(x)
        return x

kongshan = Kong()
print(kongshan)

writer = SummaryWriter('logs')
step = 0
for data in dataloader:
    img,target = data
    output = kongshan(img)
    print(img.shape)
    print(output.shape)
    writer.add_image('max_pool',output,global_step=step,dataformats='NCHW')
    step += 1
writer.close()




import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader

test_data = torchvision.datasets.CIFAR10(root='./data',train = False,download = True,transform=transforms.ToTensor())

test_loader = DataLoader(dataset = test_data,batch_size= 64,shuffle = True,num_workers = 0)

writer = SummaryWriter('logs')
for epoch in range(3):
    step = 0
    for img,target in test_loader:
        data = img
        writer.add_image('loader{}'.format(epoch),data,global_step=step,dataformats='NCHW')
        step += 1
writer.close()

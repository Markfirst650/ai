import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

train_dataset = torchvision.datasets.CIFAR10(root='./data',train =True,download = True,transform = transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10(root='./data',train =False,download = True,transform = transforms.ToTensor())

# i = 0
# for img,label in train_dataset:
#     print(img,label)
#     i += 1
# print(i)

#太多了，足足有五万个数据，下面我只选取10给数据
writer = SummaryWriter("logs")

for i in range(10):
    img,label = train_dataset[i]
    writer.add_image('train_img',img,i)
writer.close()

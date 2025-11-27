import torchvision
from torch import nn
from torchvision import transforms
from torchvision.models import VGG16_Weights

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

model = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)

#CIFAR10 是10个输出类，但vgg16足足有1000个，所以要对vgg16经行修改

#添加
# model.add_module('add_linear',nn.Linear(1000, 10))
#在指定层添加
# model.classifier.add_module('add_linear',nn.Linear(in_features=4096, out_features=10))
#修改指定内容
model.classifier[6]=nn.Linear(in_features=4096, out_features=10)
print(model)

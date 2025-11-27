import torch
import torchvision
from torch.distributed.checkpoint import state_dict
from torchvision.models import VGG16_Weights

model = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)

#保存方式1   框架+参数
torch.save(model,'vgg16_0.pth')
#加载
model1 = torch.load('vgg16_0.pth')
print(model1)

#保存方式2   只保存参数（权重）
torch.save(model.state_dict(),'vgg16_1.pth')
#加载
model2 = torchvision.models.vgg16(pretrained=False)
model2.load_state_dict(torch.load('vgg16_1.pth'))
print(model2)


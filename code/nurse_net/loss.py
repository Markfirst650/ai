#设置几个值来体验一下
import torch
from torch import nn

inputs = torch.tensor([0.485, 0.456, 0.406])
outputs = torch.tensor([0.229, 0.224, 0.225])

# loss = nn.L1Loss(reduction='sum')
loss = nn.CrossEntropyLoss()
result = loss(outputs, inputs)

result.backward()


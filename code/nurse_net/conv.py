import torch
import torch.nn.functional as F

img = torch.tensor([[1,2,0,3,1],
                   [0,1,2,3,1],
                   [1,2,1,0,0],
                   [5,2,3,1,1],
                   [2,1,0,1,1]])

kernel = torch.tensor([[1,2,1],[0,1,0],[2,1,0]])
print(img.shape)
print(kernel.shape)
img = torch.reshape(img, (1,1,5,5))
kernel = torch.reshape(kernel,(1,1,3,3))
print(img.shape)
print(kernel.shape)
output =F.conv2d(img,kernel,stride = 1)
print(output)
import torch
import torchvision
from PIL import Image
from torch import nn, relu
from torch.nn.functional import log_softmax
from torchvision import transforms


class Net (nn.Module):
    def __init__(self):
        super().__init__()
        #全连接层只接受展平的数据输入
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
model = torch.load('num_detact.pth')
model = model.cuda()
model.eval()
with torch.no_grad():

    input_data = Image.open(r"D:\Users\Jay\Downloads\02f0e705fcf0f10aba8238b7c95ba928.png")
    input_data = torchvision.transforms.Compose([transforms.Resize((28,28)),transforms.Grayscale(1)
                                                 ,transforms.ToTensor()])(input_data)
    input_data = input_data.cuda()
    output = model(input_data.view(-1,28*28))
    result = torch.argmax(output)
    print(result)
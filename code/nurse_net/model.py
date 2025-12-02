import torch
from torch import nn


#搭建神经网络
class Net(nn.Module):#-----M要大写
    def __init__(self):
        super().__init__() #super(Net,self).__init__()
        # self.conv1 = nn.Conv2d(in_channels = 3,out_channels = 32,kernel_size = 5)
        #用sequential写 ，后面定义方法更简单
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(kernel_size=2),#------M 和 P 都是大写
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),#把多维数据展平，变成一维
            nn.Linear(4*4*64,64),
            nn.Linear(64,10)
        )#-----大写

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':#-----用于测试，且被调用时不会执行
    model = Net()
    input = torch.ones(64,3,32,32)
    output = model(input)
    print(output.shape)

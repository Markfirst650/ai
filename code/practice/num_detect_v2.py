import torch
import torchvision

from torch import nn
import torch.nn.functional as F
from torch.cuda.tunable import tuning_enable
from torch.utils.data import DataLoader


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,3,padding=1)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)

        self.pool = nn.MaxPool2d(2,2)
        self.adapt = nn.AdaptiveAvgPool2d((7,7))

        self.fc1 = nn.Linear(64*7*7,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = self.adapt(x)
        x = torch.flatten(x,1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_data_loader(is_train):#需要传入的is_train来决定是训练数据集还是测试数据集
    data = torchvision.datasets.MNIST(root='./data',train=is_train,transform=torchvision.transforms.ToTensor(),download=True)
    loader = DataLoader(data,batch_size=64,shuffle=True)
    return loader

def evaluate(test_data,net):#需要传入测试数据集和网络
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for img ,target in test_data:
            if torch.cuda.is_available():
                img = img.cuda()
                target = target.cuda()

            output = net(img)#CNN 需要[batch,channel,h,w]格式的 img正好就是
            pred = output.argmax(1)
            correct += (pred == target).sum().item()
            # total = len(test_data) 这算的是领队数，因为是dataloader加载好的了
            total += target.size(0)#算的是总的人头数
            #size() 查看tensor张量维度大小的函数
            #tensor的格式[batchsize,chanel,h,w]
            #（0）就是查看第一维度的大小，也就是batchsize
            #把每个batch的大小全都加起来，也就是总的人头数了，即总的图片数
    return correct / total#返回测试的正确率

def main():#主函数，即包含训练功能以及调用加载数据集，测试函数
    net = Net()
    if torch.cuda.is_available():
        net = net.cuda()
    test_data = get_data_loader(False)
    train_data = get_data_loader(True)

    initial_accuracy = evaluate(test_data,net)
    print(f'------初始的正确率为：{initial_accuracy}-------')

    optimizer = torch.optim.Adam(net.parameters(),lr=0.001)
    for epoch in range(50):
        net.train()
        running_loss = 0.0
        for img,target in train_data:
            if torch.cuda.is_available():
                img = img.cuda()
                target = target.cuda()
            optimizer.zero_grad()#梯度清零 放在optimizer.step()之前就行了
            output = net(img)
            loss = F.cross_entropy(output,target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        current_accuracy = evaluate(test_data,net)
        print(f'第{epoch+1}轮完成，Loss为：{running_loss}，正确率为：{current_accuracy}')

    torch.save(net.state_dict(),'num_detect_v2.pth')
if __name__ =='__main__':
    main()





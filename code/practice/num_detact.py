import torch
import torchvision
from torch import nn, argmax, cuda
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.nn.functional import relu, log_softmax
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


#构建网络框架
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

#获取和加载数据集
def get_data_loader(train):#传入一个决定是训练集还是测试集的参数
    data = torchvision.datasets.MNIST(root='./data',train=True,transform = transforms.Compose([transforms.ToTensor()]), download=True)
    return DataLoader(data, batch_size=64, shuffle=True)

#测试函数，用于测试训练的效果
def evaluate(test_data,net):
    accuracy = 0
    total = 0
    with torch.no_grad():
        for(img,target)in test_data:#取的1个是批次，一个批次有64张图片
            if cuda.is_available():
                img = img.cuda()
                target = target.cuda()
            #img是28*28的像素组成的，但网络模型只允许展平了的一维输入
            #使用x.view()改变形状
            img = img.view(-1,28*28)#-1的意思是，让它自己根据图片原始像素大小以及要求的输出参数，自动确定维数
            outputs = net(img)
            #再来算正确率
            for i ,output in enumerate(outputs):
                if argmax(output) == target[i]:
                    accuracy += 1
                total += 1#记录又多少个数据被测试，放在if外，不以if条件是否成立而改变
    return accuracy / total

#主函数
def main():
    train_data = get_data_loader(True)
    test_data = get_data_loader(False)#len(test_data)---->有多少个批次
    net = Net()
    net = net.cuda()
    writer = SummaryWriter('logs')
    print('初始正确率是: {}'.format(evaluate(test_data,net)))
    optimizer = torch.optim.Adam(net.parameters(),lr=0.001)#net.parameters选择优化对象为net网络里的参数
    loss_func = nn.NLLLoss()
    if cuda.is_available():
        loss_func.cuda()
    #开始训练！！！
    for epoch in range(1):
        net.train()
        for img,target in train_data:
            if cuda.is_available():
                img = img.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            img = img.view(-1,28*28)
            output = net(img)
            loss = loss_func(output,target)
            loss.backward()
            optimizer.step()
        print('第{}轮的正确率是：{}'.format(epoch+1,evaluate(test_data,net)))
        writer.add_scalar('正确率',evaluate(test_data,net),epoch+1)



    torch.save(net,'num_detect1.pth')
    torch.save(net.state_dict(),'num_detect2.pth')



    print('模型已保存')
    writer.close()

if __name__ == '__main__':
        main()

























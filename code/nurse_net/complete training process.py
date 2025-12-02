#准备数据集 需要使用到官方提供的数据集，先引入torch vision
import torchvision

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
#训练数据
train_dataset = torchvision.datasets.CIFAR10(root = 'data',train = True,download=True,transform=torchvision.transforms.ToTensor())
#测试数据
test_dataset = torchvision.datasets.CIFAR10(root='data',train=False,download=True,transform=torchvision.transforms.ToTensor())

#获得数据集长度
train_dataset_size = len(train_dataset)
test_dataset_size = len(test_dataset)
print('训练数据集的长度为: {}'.format(train_dataset_size))
print('测试数据集的长度为: {}'.format(test_dataset_size))

#数据集打包
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=64,shuffle=True)

#创建网络模型
model = Net()
if torch.cuda.is_available():
    model = model.cuda()

#损失函数
loss_fn = nn.CrossEntropyLoss()#-----交叉熵-----
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

#优化器
learning_rate = 0.01 #-----1e-2
optimer = torch.optim.SGD(model.parameters(), lr=learning_rate)#---随机梯度下降

#设置训练网络的参数
# 记录训练的步数
total_train_step = 0
total_test_step = 0
#训练的轮数
epoch = 100

#tensorboard可视化
writer = SummaryWriter()
for i in range(epoch):
    print('---------第{}轮训练开始----------'.format(i+1))

    #训练开始
    model.train()
    for img ,target in train_loader:
        if torch.cuda.is_available():
            img = img.cuda()
            target = target.cuda()
        output = model(img)
        loss = loss_fn(output,target)#损失函数，输出的结果 与 预期的标签
        #优化前梯度清零
        optimer.zero_grad()
        loss.backward()
        optimer.step()#----优化----
        total_train_step += 1 #记录步数
        if total_train_step % 100 == 0:#----不需要每次都打印
            print('训练次数: {},loss: {}'.format(total_train_step,loss.item()))
            writer.add_scalar('train_loss',loss.item(),total_train_step)


    #进行测试
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():#-----测试不需要梯度
        for img ,target in test_loader:
            if torch.cuda.is_available():
                img = img.cuda()
                target = target.cuda()
            output = model(img)
            loss = loss_fn(output,target)
            total_test_loss += loss.item()
            accuracy = (output.argmax(1) == target).sum()
            total_accuracy += accuracy.item()

        print('整体测试集的loss: {}'.format(total_test_loss))#---每一轮 total_test_loss都会清零，不需要再除总数
        print('整体测试集的正确率: {}'.format(total_accuracy/test_dataset_size))
        writer.add_scalar('test_loss',total_test_loss,total_test_step)#---tota_test_step要加一
        writer.add_scalar('test_accuracy',total_accuracy/test_dataset_size,total_test_step)
        total_test_loss += 1

    #保存每一轮训练完的模型（注意缩进位置）
torch.save(model,'model_{}.pth'.format(i))
print('模型已保存')
writer.close()








import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, Flatten, Linear
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model1 = nn.Sequential(
            Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x
writer = SummaryWriter('logs')
net = Net()
net = net.cuda()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
steps = 0
running_loss1 = 0.0
for epoch in range(100):
    running_loss = 0.0
    for img ,target in dataloader:
        img = img.cuda()
        target = target.cuda()
        output = net(img)
        loss_result = loss(output, target)
        optimizer.zero_grad()
        loss_result.backward()
        optimizer.step()
        running_loss += loss_result.item()
        running_loss1 += loss_result.item()
        steps += 1
        avg_epoch_loss = running_loss / len(dataloader)

        if steps % 100 == 0:
            avg_step_loss = running_loss1/100
            writer.add_scalar('step_loss', avg_step_loss, steps)
            running_loss1 = 0.0
        writer.add_scalar('epoch_loss', avg_epoch_loss, epoch)

writer.close()





from torch.utils.tensorboard import SummaryWriter
import cv2

writer = SummaryWriter("logs")#对SummaryWriter创建实例 文件存储在logs文件夹下
image_path = r"D:\pycharm_space\PycharmProjects\PythonProject1\dataset\train\bees\21399619_3e61e5bb6f.jpg"
img = cv2.imread(image_path,cv2.IMREAD_COLOR)
#常用的两个方法
print(type(img))
print(img.shape)
writer.add_images("test",img,2,dataformats="HWC")
for i in range(100):
    writer.add_scalar('y = 3x',3*i,i)

writer.close()
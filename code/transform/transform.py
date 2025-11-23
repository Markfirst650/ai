from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = r"D:\pycharm_space\PycharmProjects\PythonProject1\transform\djsy.png"
img = Image.open(img_path)   #把图片重新存起来 Image.open读取之后存的是PIL格式的
print(type(img))
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img)
#使用tensorboard查看图片
writer = SummaryWriter("logs")
writer.add_images("test",tensor_img,dataformats = "CHW")
writer.close()
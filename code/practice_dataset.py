from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_dir = os.listdir(self.path)
    def __getitem__(self,idx):
        img_name = self.img_dir[idx]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img,label
    def __len__(self):
        return len(self.img_dir)

root_dir = "dataset/train"
ants_label_dir = "ants"
ants_data = MyData(root_dir,ants_label_dir)
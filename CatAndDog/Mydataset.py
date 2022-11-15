# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 14:01:08 2022

@author: pt
"""

from torch.utils.data import DataLoader,Dataset
import os
from PIL import Image
import torch
import numpy as np
from torchvision.transforms import ToPILImage,ToTensor

class MyDataset(Dataset):

    mean = [0.4870, 0.4537, 0.4161]
    std = [0.2624, 0.2558, 0.2580]

    def __init__(self,path):
        self.path = path
        self.dataset = os.listdir(path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        name = self.dataset[item]
        img = Image.open(os.path.join(self.path,name))
        # img = ToTensor()(img)
        img = np.array(img) / 255.
        img = (img - MyDataset.mean) / MyDataset.std
        img = torch.tensor(img, dtype=torch.float32).permute(2,0,1)
        target = int(name.split(".")[0])
        target = torch.Tensor([target])
        return img , target
if __name__ == '__main__':
    data = MyDataset(r"cat_dog\img")
    loader = DataLoader(dataset=data, batch_size=12000, shuffle=True)
    dataloader= next(iter(loader))[0]
    mean = torch.mean(dataloader, dim=(0,2,3))
    std = torch.std(dataloader, dim=(0,2,3))
    # 计算数据集的均值方差
    print(mean ,std)
   
    # 显示图片
    x = data[0][0]
    # x:chw 3 100 100  mean:(3,)
    x = (x * torch.tensor(MyDataset.std, dtype=torch.float32).reshape(3,1,1) + torch.tensor(MyDataset.mean,dtype=torch.float32).reshape(3,1,1))
    x = ToPILImage()(x)
    print(x)
    x.show()
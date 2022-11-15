# -*- encoding : utf-8 -*-
"""
@project = DeepLearningProject
@file = Test
@author = wly
@create_time = 2022/10/23 15:48
"""
import torch
from Mydataset import MyDataset
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as trans
import matplotlib.pyplot as plt

def test():
    mydataset = MyDataset(r".\cat_dog\test")
    dataloader = DataLoader(dataset=mydataset, batch_size=256, shuffle=True)
    net = torch.load(r".\models\net.pth")
    for x, y in dataloader:
        datas = (x * torch.tensor(MyDataset.std, dtype=torch.float32).reshape(3, 1, 1)
                 + torch.tensor(MyDataset.mean, dtype=torch.float32).reshape(3, 1, 1))
        datas = datas.numpy().transpose(0, 2, 3, 1)
        for i in range(x.shape[0]):
            data = datas[i]
            out = net(x[i])
            predict = out.argmax(1)
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.imshow(data)
            plt.subplot(1, 2, 2)
            if predict.item() == 0:
                plt.text(0.4, 0.45, "cat", fontsize=20)
            else:
                plt.text(0.4, 0.45, "dog", fontsize=20)
            plt.pause(1)

test()

def testpic():
    img = Image.open(r".\cat_dog\ourdata\14.jpg")
    img = img.resize((100, 100), Image.ANTIALIAS)
    img = trans.ToTensor()(img)
    img = trans.Normalize(MyDataset.mean, MyDataset.std)(img)
    img = img.unsqueeze(0)
    net = torch.load(r".\models\net.pth")
    out = net(img)
    print(out.max().data)
    print("猫" if out.argmax(1).item() == 0 else "狗")

testpic()
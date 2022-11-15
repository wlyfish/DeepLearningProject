# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 14:02:18 2022

@author: pt
"""

from Mynet import MLPNet
import torch
import torch.nn as nn
from Mydataset import MyDataset
from torch.utils import data
import torch.nn.functional as F
class Trainer:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = MLPNet().to(self.device)
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters())

    def train(self):
        BATCH_SIZE = 400
        NUM_EPOCHS = 20
        mydataset = MyDataset(r".\cat_dog\img")
        dataloader = data.DataLoader(dataset=mydataset,batch_size=BATCH_SIZE,shuffle=True)

        for epochs in range(NUM_EPOCHS):
            for i ,(x,y) in enumerate(dataloader):
                x = x.to(self.device)
                y = y.to(self.device)
                out = self.net(x)
                y = F.one_hot(y.long()).float()
                y = y.squeeze(1)  # 删掉一个轴，因为y.shape=[400,1,2],1是因为在做数据集的时候加了一个中括号[target]
                loss = self.loss_func(out, y)

                if i % 10 == 0:
                    print("epochs:[{}]/[{}],iteration:[{}]/[{}],loss:{}".format(epochs,NUM_EPOCHS,i, len(dataloader), loss.float()))
                    accuracy = torch.mean((out.argmax(1) == y.argmax(1)), dtype=torch.float32)  # 布尔值转float
                    # [11111100]  == 6/85
                    print("accuracy:{}".format(str(round(accuracy.item() * 100))) + "%")

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            torch.save(self.net, r"models\net.pth")

if __name__ == '__main__':

    t = Trainer()
    t.train()
# -*- encoding : utf-8 -*-
"""
@project = DeepLearningProject
@file = LogisticRegression
@author = wly
@create_time = 2022/9/22 20:19
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 数据条目
n_item = 1000
# 特征维数
n_feature = 2
# 学习率
learning_rate = 0.001
# 训练轮数
epochs = 100

# fake data
torch.manual_seed(123)
data_X = torch.randn(size=(n_item, n_feature)).float()
data_y = torch.where(torch.subtract(data_X[:, 0] * 0.5, data_X[:, 1] * 1.5) > 0, 1., 0).float()

# print(data_X.shape)

# plt.scatter(data_X[data_y == 0, 0], data_X[data_y == 0, 1])
# plt.scatter(data_X[data_y == 1, 0], data_X[data_y == 1, 1])
# plt.show()

class LogisticRegressionManually(object):
    def __init__(self):
        # regression model
        self.w = torch.randn(size=(n_feature, 1), requires_grad=True)
        self.b = torch.zeros(size=(1, 1), requires_grad=True)

    def forward(self, x):
        # y_hat = F.sigmoid(torch.matmul(self.w.transpose(0, 1), x) + self.b)
        y_hat = torch.sigmoid(torch.matmul(self.w.transpose(0, 1), x) + self.b)
        return y_hat

    @staticmethod
    def loss_func(y_hat, y):
        return -(torch.log(y_hat)*y + (1 - y)*torch.log(1 - y_hat))

    # train
    def train(self):
        for epoch in range(epochs):
            # 1. load data
            for step in range(n_item):
                # 2. forward calc
                y_hat = self.forward(data_X[step])
                y = data_y[step]    # target
                # 3. loss calc
                loss = self.loss_func(y_hat, y)
                # 4. backward
                loss.backward()
                # 5. update model(update param)
                with torch.no_grad():
                    self.w.data -= learning_rate * self.w.grad.data
                    self.b.data -= learning_rate * self.b.grad.data
                self.w.grad.data.zero_()
                self.b.grad.data.zero_()
            print('Epoch: %03d, Loss: %.3f' % (epoch, loss.item()))

if __name__ == '__main__':
    # 新建模型
    lrm = LogisticRegressionManually()
    # 训练
    lrm.train()


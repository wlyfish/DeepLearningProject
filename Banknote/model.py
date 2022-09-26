# -*- encoding : utf-8 -*-
"""
@project = DeepLearningProject
@file = model
@author = wly
@create_time = 2022/9/24 18:02
"""
import torch
from torch import nn
from torch.nn import functional as F
from config import HP


class BanknoteClassificationModel(nn.Module):
    def __init__(self,):
        super(BanknoteClassificationModel, self).__init__()
        self.linear_layer = nn.ModuleList([
            nn.Linear(in_features=in_dim, out_features=out_dim)
            for in_dim, out_dim in zip(HP.layer_list[:-1], HP.layer_list[1:])
        ])

    def forward(self, input_x):
        for layer in self.linear_layer:
            input_x = layer(input_x)
            input_x = torch.relu(input_x)
        return input_x


# if __name__ == '__main__':
#     model = BanknoteClassificationModel().to(HP.device)
#     x = torch.randn(size=(16, HP.in_features)).to(HP.device)
#     y_pred = model(x)
#     print(x)
#     print(y_pred)
#     print(y_pred.size())


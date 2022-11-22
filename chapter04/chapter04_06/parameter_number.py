# -*- encoding : utf-8 -*-
"""
@project = DeepLearningProject
@file = parameter_number
@author = wly
@create_time = 2022/11/20 20:22
"""
import torch
from torch import nn

# 统计参数个数
def model_param_number_calc(model_):
    return sum([p.numel() for p in model_.parameters() if p.requires_grad])

# input:[10, 10, 3] -> output:[10, 10 30]

# 全连接神经网络 fully-connected nn
model_fc = nn.Linear(in_features=10*10*3, out_features=10*10*30)
print('fc', model_param_number_calc(model_fc))

# 最基本的二维卷积 basic conv3d
model_basic_conv2d = nn.Conv2d(in_channels=3, out_channels=30, kernel_size=(10, 10), bias=True)
print('basic_conv2d', model_param_number_calc(model_basic_conv2d))

# 空洞卷积 dilated conv2d
model_dilated_conv2d = nn.Conv2d(in_channels=3, out_channels=30, kernel_size=(10, 10), bias=True, dilation=(2, 2))
print('model_dilated_conv2d', model_param_number_calc(model_dilated_conv2d))

# 分组卷积 group conv2d
model_group_conv2d = nn.Conv2d(in_channels=3, out_channels=30, kernel_size=(10, 10), bias=True, groups=3)
print('model_group_conv2d', model_param_number_calc(model_group_conv2d))

# 点卷积 point-wise conv2d
model_pointwise_conv2d = nn.Conv2d(in_channels=3, out_channels=30, kernel_size=(1, 1), bias=True)
print('model_pointwise_conv2d', model_param_number_calc(model_pointwise_conv2d))

# 深度可分离卷积 deep separable conv2d
depth_conv2d = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(10, 10), groups=3)
point_conv2d = nn.Conv2d(in_channels=3, out_channels=30, kernel_size=(1, 1))
print('model_ds_conv2d', model_param_number_calc(depth_conv2d) + model_param_number_calc(point_conv2d))

# transpose conv2d
transpose_conv2d = nn.ConvTranspose2d(in_channels=3, out_channels=30, kernel_size=(10, 10))
print(transpose_conv2d(torch.randn(size=(1, 3, 10, 10))).size())

# -*- encoding : utf-8 -*-
"""
@project = DeepLearningProject
@file = demo_create
@author = wly
@create_time = 2022/9/19 9:27
"""
import torch
import numpy as np

a = torch.Tensor([[1, 2], [3, 4]])
print(a)
print(a.type())

a = torch.Tensor(2, 3)
print(a)
print(a.type())

a = torch.ones(2, 2)
print(a)
print(a.type())

a = torch.eye(2, 2)
print(a)
print(a.type())

a = torch.zeros(2, 2)
print(a)
print(a.type())

b = torch.Tensor(3, 4)
c = torch.zeros_like(b)
c = torch.ones_like(b)
print(c)
print(c.type())

# 随机
d = torch.rand(3, 4)
print(d)
print(d.type())

d = torch.normal(mean=0.0, std=torch.rand(5))
print(d)
print(d.type())

d = torch.normal(mean=torch.rand(5), std=torch.rand(5))
print(d)
print(d.type())

d = torch.Tensor(2, 2).uniform_(-1, 1)
print(d)
print(d.type())

# 序列
a = torch.arange(0, 10, 1)
print(a)
print(a.type())

# 拿到等间隔的n个数字
a = torch.linspace(2, 10, 4)
print(a)
print(a.type())

a = torch.randperm(10)
print(a)
print(a.type())

######################
a = np.array([[1, 2], [3, 4]])
print(a)
print(a.dtype)

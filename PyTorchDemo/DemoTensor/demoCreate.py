# -*- encoding : utf-8 -*-
"""
@project = DeepLearningProject
@file = demoCreate
@author = wly
@create_time = 2022/10/28 16:15
"""
import torch

a = torch.Tensor([[1, 2], [3, 4]])
print(a)

# tensor([[1., 2.],
#         [3., 4.]])

print(a.type())  # torch.FloatTensor

b = torch.FloatTensor(2, 3)
print(b)
print(b.type())

'''几种特殊的Tensor'''
c = torch.ones(2, 3)
print(c)
print(c.type())

d = torch.zeros(2, 3)
print(d)
print(d.type())

e = torch.eye(4, 4)
print(e)
print(e.type())

f = torch.zeros_like(a)
print(f)
print(f.type())

'''随机'''
a = torch.rand(2, 4)
print(a)  # 0 - 1之间的随机值
print(a.type())

b = torch.normal(mean=0, std=torch.rand(5))
print(b)
print(b.type())

# 均匀分布
a = torch.Tensor(2, 2).uniform_(-1, 1)
print(a)
print(a.type())

'''序列'''
a = torch.arange(0, 10, 3)  # torch.LongTensor
print(a)
print(a.type())

a = torch.linspace(2, 10, 4)  # 拿到等间隔的n个数字
print(a)
print(a.type())

a = torch.randperm(10)  # torch.LongTensor
print(a)
print(a.type())


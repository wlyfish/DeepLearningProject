# -*- encoding : utf-8 -*-
"""
@project = DeepLearningProject
@file = demo_statistic
@author = wly
@create_time = 2022/9/26 22:15
"""
import torch

a = torch.rand(2, 2)

print(a)
print(torch.mean(a, dim=0))
print(torch.sum(a, dim=0))
print(torch.prod(a, dim=0))

print(torch.argmax(a, dim=0))
print(torch.argmin(a, dim=0))

print(torch.median(a))
print(torch.std(a))
print(torch.var(a))
print(torch.mode(a))

print("######### 直方图 ###########")
a = torch.rand(2, 3)*10
print(a)
print(torch.histc(a, 6, 0, 0))
print(torch.sum(torch.histc(a, 6, 0, 0)))

a = torch.randint(0, 10, [10])
print(a)
print(torch.bincount(a))


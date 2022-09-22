# -*- encoding : utf-8 -*-
"""
@project = DeepLearningProject
@file = demo_op_2
@author = wly
@create_time = 2022/9/21 20:24
"""
import torch

a = torch.rand(2, 2)
a = a * 10

print(a)

print(torch.floor(a))
print(torch.ceil(a))
print(torch.round(a))
print(torch.trunc(a))
print(torch.frac(a))
print(a % 2)

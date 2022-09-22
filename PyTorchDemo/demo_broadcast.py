# -*- encoding : utf-8 -*-
"""
@project = DeepLearningProject
@file = demo_broadcast
@author = wly
@create_time = 2022/9/21 11:06
"""
import torch

a = torch.rand(2, 1, 1, 3)
b = torch.rand(3)
c = a + b
print(a)
print(b)
print(c)
print(c.shape)


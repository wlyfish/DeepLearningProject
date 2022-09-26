# -*- encoding : utf-8 -*-
"""
@project = DeepLearningProject
@file = demo_trigon
@author = wly
@create_time = 2022/9/26 9:14
"""
import torch

a = torch.rand(2, 3)
b = torch.cos(a)

print(a)
print(b)

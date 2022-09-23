# -*- encoding : utf-8 -*-
"""
@project = DeepLearningProject
@file = demo_backward_01
@author = wly
@create_time = 2022/9/23 15:17
"""
import torch

a = torch.tensor(10., requires_grad=True)
b = torch.tensor(20., requires_grad=True)

F = a * b

# calculate the gradients
F.backward()

print('a = {}'.format(a.grad))
print('b = {}'.format(b.grad))


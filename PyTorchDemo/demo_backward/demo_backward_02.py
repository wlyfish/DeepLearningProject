# -*- encoding : utf-8 -*-
"""
@project = DeepLearningProject
@file = demo_backward_02
@author = wly
@create_time = 2022/9/23 15:29
"""
import torch

a = torch.tensor([10., 10.], requires_grad=True)
b = torch.tensor([20., 20.], requires_grad=True)

F = a * b

F.backward(gradient=torch.tensor([1., 1.]))

print('a = {}'.format(a.grad))
print('b = {}'.format(b.grad))

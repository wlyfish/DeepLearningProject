# -*- encoding : utf-8 -*-
"""
@project = DeepLearningProject
@file = demo_backward_03
@author = wly
@create_time = 2022/9/23 16:02
"""
import torch

alpha = torch.tensor([10.0, 100.0, 1000.0])

X = torch.tensor([4.0, 3.0, 2.0], requires_grad=True)
Y = torch.zeros(3)

x0, x1, x2 = X[0], X[1], X[2]
y0 = 3*x0 + 7*x1**2 + 6*x2**3
y1 = 4*x0 + 8*x1**2 + 3*x2**3
y2 = 5*x0 + 9*x1**2 + 1*x2**3

print('X = {}'.format(X))
print('Y = {}'.format(Y))

Y[0], Y[1], Y[2] = y0, y1, y2
print('Y = {}'.format(Y))

print('X.grad = {}'.format(X.grad))
print('Y.grad = {}'.format(Y.grad))

Y.backward(gradient=alpha)
print('Y.grad = {}'.format(Y.grad))
print('X.grad = {}'.format(X.grad))

# -*- encoding : utf-8 -*-
"""
@project = DeepLearningProject
@file = demo_compare
@author = wly
@create_time = 2022/9/21 20:48
"""
import torch
import numpy as np


a = torch.rand(2, 3)
b = torch.rand(2, 3)

print(a)
print(b)

print(torch.eq(a, b))
print(torch.equal(a, b))
print(torch.ge(a, b))
print(torch.le(a, b))
print(torch.gt(a, b))
print(torch.lt(a, b))
print(torch.ne(a, b))

##########
a = torch.tensor([1, 4, 4, 3, 5])
print(a.shape)

# 默认升序
print(torch.sort(a))
print(torch.sort(a, dim=0, descending=True))

print("############# sort ###############")

b = torch.tensor([[1, 4, 4, 3, 5],
                  [2, 3, 1, 3, 5]])
print(b)
print(b.shape)
print(torch.sort(b, dim=1, descending=False))

print("############# topk ###############")
a = torch.tensor([[2, 4, 3, 1, 5],
                  [2, 3, 5, 1, 4]])
print(a)
print(torch.topk(a, k=2, dim=1))

print(torch.kthvalue(a, k=2, dim=0))
print(torch.kthvalue(a, k=2, dim=1))

a = torch.rand(2, 3)
print(a)
print('a.isfinite = {}'.format(torch.isfinite(a)))
print('a/0 isfinite = {}'.format(torch.isfinite(a/0)))
print('a/0 isinf = {}'.format(torch.isinf(a/0)))
print('a isnan = {}'.format(torch.isnan(a)))

a = torch.tensor([1, 2, np.nan])
print(a)
print('a isnan = {}'.format(torch.isnan(a)))

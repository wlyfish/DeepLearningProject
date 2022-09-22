# -*- encoding : utf-8 -*-
"""
@project = DeepLearningProject
@file = demo_compare
@author = wly
@create_time = 2022/9/21 20:48
"""
import torch

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

print("############################")

b = torch.tensor([[1, 4, 4, 3, 5],
                  [2, 3, 1, 3, 5]])
print(b)
print(b.shape)
print(torch.sort(b, dim=0, descending=False))

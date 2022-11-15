# -*- encoding : utf-8 -*-
"""
@project = DeepLearningProject
@file = demo_op
@author = wly
@create_time = 2022/11/6 15:42
"""
import torch

a = torch.rand(2, 3)
b = torch.rand(2, 3)

print(a)
print(b)

# add
print("==== add============")
print(a + b)
print(a.add(b))
print(torch.add(a, b))
print(a.add_(b))
print(a)

# sub
print("==== sub============")
print(a - b)
print(a.sub(b))
print(torch.sub(a, b))
print(a.sub_(b))

print("===== matmul =================")
a = torch.ones(2, 1)
b = torch.ones(1, 2)

print(a.mul(b))
print(torch.mul(a, b))
print(a @ b)
print(a.mm(b))
print(torch.mm(a, b))

# 高维tensor
a = torch.ones(1, 2, 3, 4)
b = torch.ones(1, 2, 4, 3)
print(a)
print(torch.matmul(a, b))

# pow
a = torch.tensor([1, 2])
print(a**3)
print(torch.pow(a, 3))
print(a.pow(3))
print(a. pow_(3))

# exp
a = torch.tensor([1, 2], dtype=torch.float32)
print(a.type())
print(torch.exp(a))
print(a.exp())
print(a.exp_())
print(torch.exp_(a))

# log
a = torch.tensor([1, 2], dtype=torch.float32)
print(torch.log(a))
print(a.log())
print(torch.log_(a))



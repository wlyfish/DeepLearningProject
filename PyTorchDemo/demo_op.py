# -*- encoding : utf-8 -*-
"""
@project = DeepLearningProject
@file = demo_op
@author = wly
@create_time = 2022/9/20 20:29
"""
import torch

a = torch.rand(2, 3)
b = torch.rand(2, 3)

# add
print("a =", a)
print("b =", b)

print("a + b =", a + b)
print("a.add(b) =", a.add(b))
print("torch.add(a, b) =", torch.add(a, b))
print("a.add_(b) =", a.add_(b))
print("a =", a)

# sub
print("a - b =", a - b)
print("a.sub(b) =", a.sub(b))
print("torch.sub(a, b) =", torch.sub(a, b))
print("a.sub_(b) =", a.sub_(b))

# mul
print("===== mul =====")
print(a * b)
print(torch.mul(a, b))
print(a.mul(b))
print(a)
print(a.mul_(b))
print(a)

# div
print("===== div =====")
print(a / b)
print(torch.div(a, b))
print(a.div(b))
print(a.div_(b))

# matmul 矩阵运算
a = torch.ones(2, 1)
b = torch.ones(1, 2)
# print(a)
# print(b)
print(a @ b)
print(a.matmul(b))
print(a.mm(b))
print(torch.matmul(a, b))

# 高维tensor
a = torch.ones(1, 2, 3, 4)
b = torch.ones(1, 2, 4, 3)
print(a)
print(b)
print(a.matmul(b).shape)

# pow
a = torch.tensor([1, 2])
print(torch.pow(a, 3))
print(a.pow(3))
print(a**3)
print(a.pow_(3))
print(a)


# exp
print("===== exp =====")
a = torch.tensor([1, 2],
                 dtype=torch.float32)
print(a.type())
print(torch.exp(a))
print(torch.exp_(a))
print(a.exp())
print(a.exp_())

# log
a = torch.tensor([1, 2],
                 dtype=torch.float32)
print(torch.log(a))
print(torch.log_(a))

print(a.log())
print(a.log_())

# sqrt
a = torch.tensor([1, 2],
                 dtype=torch.float32)
print(torch.sqrt(a))
print(torch.sqrt_(a))

print(a.sqrt())
print(a.sqrt_())

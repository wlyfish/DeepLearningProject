# -*- encoding : utf-8 -*-
"""
@project = DeepLearningProject
@file = 非标量反向传播
@author = wly
@create_time = 2022/10/9 23:31
"""
import torch

# 定义叶子节点张量x，形状为1*2
x = torch.tensor([[2, 3]], dtype=torch.float, requires_grad=True)
# 初始化 Jacobian 矩阵
J = torch.zeros(2, 2)
# 初始化目标张量，形状为1*2
y = torch.zeros(1, 2)

# 定义y与x之间的映射关系
y[0, 0] = x[0, 0] ** 2 + 3 * x[0, 1]
y[0, 1] = x[0, 1] ** 2 + 2 * x[0, 0]

# y.backward(torch.Tensor([[1, 1]]))
# print(x.grad) # tensor([[6., 9.]])

# 生成y1对x的梯度
y.backward(torch.Tensor([[1, 0]]), retain_graph=True)
J[0] = x.grad

# 梯度是累加的，故需要对x梯度清零
x.grad = torch.zeros_like(x.grad)

y.backward(torch.Tensor([[0, 1]]))
J[1] = x.grad

print(J)

'''tensor([[4., 3.],
        [2., 6.]])'''
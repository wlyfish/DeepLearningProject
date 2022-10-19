# -*- encoding : utf-8 -*-
"""
@project = DeepLearningProject
@file = 标量反向传播
@author = wly
@create_time = 2022/10/9 22:49
"""

import torch

# 定义叶子节点及算子节点
x = torch.Tensor([2])
# 初始化权重参数W，偏移量b、并设置require_grad属性为True，为自动求导
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# 实现向前传播
y = torch.mul(w, x)
z = torch.add(y, b)

# 查看x，w，b叶子节点的requires_grad属性
print("x, w, b的require_grad属性分别为：{}, {}, {}".format(x.requires_grad, w.requires_grad, b.requires_grad))
# x, w, b的require_grad属性分别为：False, True, True

# 查看叶子节点、非叶子节点的其他属性
print("y, z的requires_grad属性分别为：{}, {}".format(y.requires_grad, z.requires_grad))
# y, z的requires_grad属性分别为：True, True

# 查看各节点是否为叶子节点
print("x, w, b, y, z的是否为叶子节点：{}, {}, {}, {}, {}".format(x.is_leaf, w.is_leaf, b.is_leaf, y.is_leaf, z.is_leaf))
# x, w, b, y, z的是否为叶子节点：True, True, True, False, False

# 查看叶子结点的grad_fn属性
print("x, w, b的grad_fun属性：{}, {}, {}".format(x.grad_fn, w.grad_fn, b.grad_fn))
# x, w, b的grad_fun属性：None, None, None

# 查看非叶子节点的grad_fn属性
print("y, z的grad_fn属性：{}, {}".format(y.grad_fn, z.grad_fn))
# y, z的grad_fn属性：<MulBackward0 object at 0x000001E854384190>, <AddBackward0 object at 0x000001E854384280>

# 自动求导，实现梯度方向传播,即梯度的方向传播
# 基于z张量的反向传播，执行backward之后计算图会自动清空
z.backward()
# 如果需要多次使用backward，需要修改retain_graph为True，此时梯度是累加的

# 查看叶子结点的梯度，x是叶子节点但它无需求导，故其梯度为None
print("参数w, b的梯度分别为：{}, {}".format(w.grad, b.grad))
# 参数w, b的梯度分别为：tensor([2.]), tensor([1.])

# 非叶子结点的梯度，执行backward后，会自动清空
print("非叶子节点y, z的梯度分别为：{}，{}".format(y.grad, z.grad))
# 非叶子节点y, z的梯度分别为：None，None


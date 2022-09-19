# -*- encoding : utf-8 -*-
"""
@project = DeepLearningProject
@file = demo_attribute
@author = wly
@create_time = 2022/9/19 15:16
"""
import torch

dev = torch.device("cpu")
dev = torch.device("cuda")

a = torch.tensor([2, 2],
                 dtype=torch.float32,
                 device=dev)
print(a)

# 坐标
i = torch.tensor([[0, 1, 2], [0, 1, 2]])
v = torch.tensor([1, 2, 3])
a = torch.sparse_coo_tensor(i, v, (4, 4),
                            dtype=torch.float32,
                            device=dev)
print(a)
print(a.to_dense())

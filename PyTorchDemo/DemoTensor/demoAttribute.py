# -*- encoding : utf-8 -*-
"""
@project = DeepLearningProject
@file = demoAttribute
@author = wly
@create_time = 2022/11/6 15:20
"""
import torch

dev = torch.device("cpu")
dev = torch.device("cuda")
a = torch.tensor([2, 2],
                 dtype=torch.float,
                 device=dev)
print(a)

# 稀疏张量
i = torch.tensor([[0, 1, 2], [0, 1, 2]])
v = torch.tensor([1, 2, 3])
a = torch.sparse_coo_tensor(i, v, (4, 4), dtype=torch.float32, device=dev)
# 转成稠密的张量
a = a.to_dense()
print(a)

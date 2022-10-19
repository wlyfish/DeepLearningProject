# -*- encoding : utf-8 -*-
"""
@project = DeepLearningProject
@file = activate_function
@author = wly
@create_time = 2022/10/19 22:48
"""

import torch
from torch import nn
import torch.nn.functional as F

layer = nn.Linear(in_features=16, out_features=5)
x = torch.randn(size=(8, 16))
layer_output = layer(x)

print(layer_output.size())

# sigmoid
layer_output = F.relu(layer_output)
print(layer_output.size())

layer_output = F.mish(layer_output)
print(layer_output.size())

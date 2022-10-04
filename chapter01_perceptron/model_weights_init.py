# -*- encoding : utf-8 -*-
"""
@project = DeepLearningProject
@file = model_weights_init
@author = wly
@create_time = 2022/10/3 22:04
"""
from torch import nn

# Xavier
model = nn.Linear(in_features=16, out_features=128)
print(model.weight)
nn.init.xavier_uniform_(model.weight, gain=nn.init.calculate_gain('tanh'))
print(model.weight)

# Kaiming
nn.init.kaiming_uniform_(model.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')
print(model.weight)

# -*- encoding : utf-8 -*-
"""
@project = DeepLearningProject
@file = model_weights
@author = wly
@create_time = 2022/10/19 21:25
"""
from torch import nn

model = nn.Linear(in_features=16, out_features=128)

print(model.weight)
nn.init.xavier_uniform_(model.weight, gain=nn.init.calculate_gain('tanh'))
print(model.weight)


nn.init.kaiming_uniform_(model.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')
print(model.weight)

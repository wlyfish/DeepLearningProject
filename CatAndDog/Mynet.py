# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 14:01:54 2022

@author: pt
"""

import torch.nn as nn
import torch

class MLPNet(nn.Module):

    def __init__(self):
        super(MLPNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(100*100*3, 128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )

    def forward(self, x):
        x = x.reshape(-1, 100*100*3)
        return self.layer(x)
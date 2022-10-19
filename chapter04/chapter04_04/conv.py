# -*- encoding : utf-8 -*-
"""
@project = DeepLearningProject
@file = conv
@author = wly
@create_time = 2022/10/19 19:15
"""
from torch import nn
from PIL import Image
from torchvision import transforms

transform = transforms.Compose(
    [transforms.ToTensor()]
)

image = Image.open('./lena.jpg')
x = transform(image)
x = x.unsqueeze(0)
batch_size, n_channels, height, width = x.size()
print('x size : ', x.size())

N_OUT_CHS = 32
KERNEL_SIZE = 11

conv2d_nn = nn.Conv2d(
    in_channels=n_channels,
    out_channels=N_OUT_CHS,
    kernel_size=(KERNEL_SIZE, KERNEL_SIZE),
    stride=1,
    padding=(KERNEL_SIZE//2, KERNEL_SIZE//2)
)

x_out = conv2d_nn(x)
print('x_out size : ', x_out.size())

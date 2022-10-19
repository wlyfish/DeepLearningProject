# -*- encoding : utf-8 -*-
"""
@project = DeepLearningProject
@file = pooling
@author = wly
@create_time = 2022/10/19 20:14
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
print('x size: ', x.size())

N_OUT_CHS = 32
KERNEL_SIZE = 11

conv2d_nn = nn.Conv2d(
    in_channels=n_channels,
    out_channels=N_OUT_CHS,
    kernel_size=(KERNEL_SIZE, KERNEL_SIZE),
    stride=1,
    padding=(KERNEL_SIZE//2, KERNEL_SIZE//2)
)

# pooling_layer = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
# x_conv2d_out = conv2d_nn(x)
# print('conv2d: ', x_conv2d_out.size())
#
# pool_out = pooling_layer(x_conv2d_out)
# print('pooling: ', pool_out.size())

# pooling_layer = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
# x_conv2d_out = conv2d_nn(x)
# print('conv2d: ', x_conv2d_out.size())
# pool_out = pooling_layer(x_conv2d_out)
# print('pooling: ', pool_out.size())

pooling_layer = nn.AdaptiveAvgPool2d(output_size=(45, 45))
x_conv2d_out = conv2d_nn(x)
print('conv2d: ', x_conv2d_out.size())
pool_out = pooling_layer(x_conv2d_out)
print('pooling: ', pool_out.size())

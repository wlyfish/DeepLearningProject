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

# 将图片转为Tensor，并进行归一化[0, 1] & to float32
transform = transforms.Compose(
    [transforms.ToTensor()]
)

image = Image.open('./lena.jpg')
x = transform(image)
x = x.unsqueeze(0)
batch_size, n_channels, height, width = x.size()
print('x size : ', x.size())

# 输出的channel， 即在这里多少个卷积核
N_OUT_CHS = 32
# 卷积核的大小，一般选奇数（有论文）
KERNEL_SIZE = 5

conv2d_nn = nn.Conv2d(
    # 3 for rgb
    in_channels=n_channels,
    out_channels=N_OUT_CHS,
    kernel_size=(KERNEL_SIZE, KERNEL_SIZE),
    # 步长
    stride=1,
    # 输入输出大小一致，方便作为下层的输入，保存边界信息
    padding=(KERNEL_SIZE//2, KERNEL_SIZE//2)
)

x_out = conv2d_nn(x)
print('x_out size : ', x_out.size())

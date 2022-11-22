# -*- encoding : utf-8 -*-
"""
@project = DeepLearningProject
@file = transform
@author = wly
@create_time = 2022/11/21 20:04
"""

import torchvision.transforms as T
from PIL import Image

trans = T.Compose(
    [
        T.ToTensor(),
        T.RandomRotation(45),
        T.RandomAffine(45),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
)

image = Image.open('./lena.jpg')
print(image)
t_out_image = trans(image)
print(t_out_image.size())
# image.show()
T.RandomRotation(90)(image).show()

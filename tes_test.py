import torchvision.transforms as transforms
import torch
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor()
])

# 1. rand 로 만들기
# # rand 로 만들기
# img = np.random.rand(256, 256, 3).astype(np.float32)
# print(img)


# 2. ByteTensor 로 변경?
img = np.random.randint(0, 256, [256, 256, 3]).astype(np.uint8)
print(img.shape)


tensor_img = transform(img)

print(tensor_img)
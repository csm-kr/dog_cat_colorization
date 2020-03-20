import os
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from skimage import color
import torch


class ColorLoader(data.Dataset):

    def __init__(self, root=None, transform=None, target_transform=None):

        if root is None:
            raise ValueError

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        name_list = os.listdir(root)
        self.images = [os.path.join(root, x) for x in name_list]
        # assert (len(self.images) == len(self.labels))

    def __getitem__(self, index):

        # 1. PIL 을 통해서 이미지 받아오기
        image = Image.open(self.images[index]).convert('RGB')

        # transform 적용
        if self.transform is not None:
            image = self.transform(image)

        image = np.asarray(image)

        # lab 로 변경
        img_lab = color.rgb2lab(image)
        img_l = img_lab[:, :, 0]
        img_l = img_l[:, :, np.newaxis]
        img_ab = img_lab[:, :, 1:3]

        # 여기서 torch 로 바꾸어준다.
        # 원래는 transforms.ToTensor()에서 .transpose((2, 0, 1))를 실행해 준다.
        # 우리는 ToTensor()를 사용하지 않기 때문에! 실제루 해 준다.

        img_l = img_l.transpose((2, 0, 1))
        img_ab = img_ab.transpose((2, 0, 1))

        # .type(torch.FloatTensor) 을 해주지 않으면 torch.DoubleTensor 라서 오류가 납니다. :)
        img_l = torch.from_numpy(img_l).type(torch.FloatTensor)
        img_ab = torch.from_numpy(img_ab).type(torch.FloatTensor)

        return img_l, img_ab

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":

    transform = transforms.Compose(
        [transforms.Resize(size=(224, 224)),
         ])
    # 얘는 img 에 대해서만 하는거임

    trainset = ColorLoader(transform=transform)
    print(len(trainset))

    # 이미지 하나씩 가져오는 부분 - 현재 .transpose((2, 0, 1)) 해서 안됨
    # for i in range(len(trainset)):
    #     img, label = trainset[i]
    #
    #     color_img = np.concatenate((img, label), axis=-1)
    #     # lab2rgb 는 0 ~ 1 사이로 나오는 것 같음.
    #     color_img = color.lab2rgb(color_img)
    #
    #     # cv2로 출력하기 위해서
    #     img = np.array(img).astype(np.uint8)
    #     label = np.array(label).astype(np.uint8)
    #     color_img *= 255
    #     color_img = color_img[50:, :, ::-1]
    #     color_img = np.array(color_img).astype(np.uint8)
    #
    #     cv2.imshow('gray_img', img)
    #     cv2.imshow('color_img', color_img)
    #     cv2.waitKey(0)

    trainloader = data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=0)

    for data, labels in trainloader:
        print(data.shape)
        print(labels.shape)






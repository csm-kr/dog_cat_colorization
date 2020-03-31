import os
import glob
from torch.utils.data import Dataset
from PIL import Image
from skimage import color
# pip install scikit-image
import numpy as np
import torch
import cv2
import torchvision.transforms as transforms


visualization = True


class ColorDataset(Dataset):
    def __init__(self, root='D:\Data\dogs-vs-cats', subset='train', transform=None):
        """
        create dog cat color dataset
        :param root: str, root path that contains test1 and train folder
        :param subset: str, goal of dataset, 'train' or 'test'
        """
        super().__init__()

        # subset must be 'train' or 'test'
        # if subset != 'train' and subset != 'test':
        #     raise Exception('Wrong input, subset must be train or test')
        # if len(subset) == 4:
        #     subset += '1'
        #
        # self.image_path = os.path.join(root, subset)
        self.image_path = root
        self.image_name = glob.glob(os.path.join(self.image_path, '*.jpg'))
        self.transform = transform

    def __getitem__(self, idx):
        # image load
        image = Image.open(self.image_name[idx])

        if self.transform is not None:
            image = self.transform(image)

        # rgb to lab
        # L lies between 0 and 100, and a and b lie between -128 to 127.
        image = np.asarray(image)  # uint 8 0 ~ 255

        img_gray = color.rgb2gray(image)  # 0 ~ 1 scale float64
        img_gray = img_gray[:, :, np.newaxis]  # h, w, 1

        img_lab = color.rgb2lab(image)   # type --> uint8 to float64
        img_lab = (img_lab + 128) / 255  # 0 ~ 1 scaling float 64

        img_ab = img_lab[:, :, 1:]       # h, w, 2

        # visualization
        if visualization:

            # lab to rgb
            img_gray_vis = img_gray * 100
            img_ab_vis = img_ab * 255 - 128

            color_img = np.concatenate((img_gray_vis, img_ab_vis), axis=-1)
            color_img = color.lab2rgb(color_img)  # rgb

            cv2.imshow('color', color_img[..., ::-1])
            cv2.imshow('gray', img_gray)
            cv2.waitKey(0)

        img_gray = img_gray.transpose((2, 0, 1))    # 1, h, w
        img_ab = img_ab.transpose((2, 0, 1))  # 2, h, w

        # .type(torch.FloatTensor) 을 해주지 않으면 torch.DoubleTensor 라서 오류가 납니다. :) <-- 어디서?
        # img_gray = torch.from_numpy(img_gray).type(torch.FloatTensor)
        # img_ab = torch.from_numpy(img_ab).type(torch.FloatTensor)
        img_gray = torch.from_numpy(img_gray)
        img_ab = torch.from_numpy(img_ab)

        return img_gray, img_ab

    def __len__(self):
        return len(self.image_name)


# test code
if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.RandomCrop(224),
    ])

    train_dataset = ColorDataset(root='D:\Data\VOC_ROOT\TEST\VOC2007\JPEGImages', subset='test', transform=transform)

    for i, (g, ab) in enumerate(train_dataset):
        print(i)
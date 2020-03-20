import os
import glob
from torch.utils.data import Dataset
from PIL import Image
from skimage import color
# pip install scikit-image
import numpy as np
import torch


class ColorDataset(Dataset):
    def __init__(self, root, subset='train'):
        """
        create dog cat color dataset
        :param root: str, root path that contains test1 and train folder
        :param subset: str, goal of dataset
        """
        super().__init__()

        # subset 이 str 이 아니면 오류
        if len(subset) != 5:
            subset += '1'
        self.image_path = os.path.join(root, subset)
        self.image_name = glob.glob(os.path.join(self.image_path, '*.jpg'))

    def __getitem__(self, idx):
        # image load
        image = Image.open(self.image_name[idx])

        # rgb to lab
        img_lab = color.rgb2lab(image)
        img_l = img_lab[:, :, 0]         # h, w
        img_l = img_l[:, :, np.newaxis]  # h, w, 1
        img_ab = img_lab[:, :, 1:]       # h, w, 2

        img_l = img_l.transpose((2, 0, 1))
        img_ab = img_ab.transpose((2, 0, 1))

        # .type(torch.FloatTensor) 을 해주지 않으면 torch.DoubleTensor 라서 오류가 납니다. :)
        img_l = torch.from_numpy(img_l).type(torch.FloatTensor)
        img_ab = torch.from_numpy(img_ab).type(torch.FloatTensor)

        return img_l, img_ab

    def __len__(self):
        return len(self.image_name)


# test code
if __name__ == "__main__":
    train_dataset = ColorDataset(root='D:\Data\dogs-vs-cats', subset='train')

    for i in train_dataset:
        print(i)
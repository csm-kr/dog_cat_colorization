import torch
import torchvision.transforms as transforms
from dataset import ColorDataset
import visdom
from torch.utils.data import DataLoader
from model import EDNet, UNet

from skimage import color
import numpy as np
import cv2

import torch
import os
import time


def test(epoch, device, vis, data_loader, model, criterion, save_path, save_file_name):

    #       # ----- load -----
    state_dict = torch.load(os.path.join(save_path, save_file_name) + '.{}.pth'.format(epoch))
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():

        tic = time.time()
        print('Epoch : {}'.format(epoch))

        for idx, (images, labels) in enumerate(data_loader):

            # ----- cuda -----
            images = images.to(device)
            labels = labels.to(device)

            # ----- loss -----
            outputs = model(images)
            loss = criterion(outputs, labels)

            # ----- eval -----
            toc = time.time() - tic

            # convert from tensor to numpy
            images = images.cpu().numpy()
            outputs = outputs.cpu().numpy()
            colored_img = labels.cpu().numpy()

            # the first image of the batches
            images = images[0]
            outputs = outputs[0]
            colored_img = colored_img[0]

            # C, H, W--> C, H, W
            images = images.transpose((1, 2, 0))
            outputs = outputs.transpose((1, 2, 0))
            colored_img = colored_img.transpose((1, 2, 0))

            # lab to rgb
            img_gray_vis = images * 100
            img_ab_vis = outputs * 255 - 128
            label = colored_img * 255 - 128

            color_img = np.concatenate((img_gray_vis, img_ab_vis), axis=-1)
            color_img = color.lab2rgb(color_img)  # rgb

            label_img = np.concatenate((img_gray_vis, label), axis=-1)
            label_img = color.lab2rgb(label_img)  # rgb

            cv2.imshow('output', color_img[..., ::-1])
            cv2.imshow('input', images)
            cv2.imshow('origin_img', label_img[..., ::-1])

            cv2.waitKey(0)

            # 3 channel 의 lab 이미지로 변환
            # color_img = np.concatenate((images, outputs), axis=-1)
            # origin_img = np.concatenate((images, colored_img), axis=-1)
            #
            # color_img = color.lab2rgb(color_img).astype(np.float32)[..., ::-1]
            # origin_img = color.lab2rgb(origin_img).astype(np.float32)[..., ::-1]
            #
            # cv2.imshow('color_img', color_img)
            # cv2.imshow('origin_img', origin_img)
            # cv2.waitKey(0)
            # ----- print -----


if __name__ == "__main__":

    epoch = 0   # FIXME
    save_path = './saves'
    save_file_name = 'unet3'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vis = visdom.Visdom()

    # 4. data set
    transform = transforms.Compose([
        transforms.Resize((256, 256))
    ])

    train_set = ColorDataset(root='D:\Data\dogs-vs-cats', subset='test', transform=transform)
    test_loader = DataLoader(dataset=train_set,
                             batch_size=2,
                             shuffle=False)

    criterion = torch.nn.MSELoss()

    model = UNet().to(device)
    test(epoch, device, vis, test_loader, model, criterion, save_path, save_file_name)






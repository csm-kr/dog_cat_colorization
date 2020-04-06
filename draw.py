import torch
from model import UNet
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from skimage import color
import cv2
import os


def draw(epoch, image, device, model, save_path, save_file_name):

    #       # ----- load -----
    state_dict = torch.load(os.path.join(save_path, save_file_name) + '.{}.pth'.format(epoch))
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():

        print('Epoch : {}'.format(epoch))

        image = np.asarray(image)  # uint 8 0 ~ 255
        img_gray = color.rgb2gray(image)  # 0 ~ 1 scale float64
        img_gray = img_gray[:, :, np.newaxis]  # h, w, 1

        img_lab = color.rgb2lab(image)  # type --> uint8 to float64
        img_lab = (img_lab + 128) / 255  # 0 ~ 1 scaling float 64
        img_ab = img_lab[:, :, 1:]  # h, w, 2

        img_gray = img_gray.transpose((2, 0, 1))  # 1, h, w
        img_ab = img_ab.transpose((2, 0, 1))  # 2, h, w

        img_gray = torch.from_numpy(img_gray).type(torch.FloatTensor)
        img_ab = torch.from_numpy(img_ab).type(torch.FloatTensor)

        img_gray = img_gray.unsqueeze(0)
        img_ab = img_ab.unsqueeze(0)

        images = img_gray.to(device)
        outputs = model(images)

        # convert from tensor to numpy
        images = images.cpu().numpy()[0]
        outputs = outputs.cpu().numpy()[0]
        colored_img = img_ab.cpu().numpy()[0]

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


if __name__ == "__main__":

    epoch = 99   # FIXME
    save_path = './saves'
    save_file_name = 'unet3'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 4. data set
    transform = transforms.Compose([
        transforms.Resize((256, 256))
    ])

    # FIXME
    # you can set the your images
    image_dir = "C:\\Users\csm81\Desktop\capture.JPG"
    image = Image.open(image_dir)
    image = transform(image)

    model = UNet().to(device)
    draw(epoch, image, device, model, save_path, save_file_name)
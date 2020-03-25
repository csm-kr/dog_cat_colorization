import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import net as network
from skimage import color
import numpy as np
import cv2


def test(opts, show=True):
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # transform 설정하기
    transform = transforms.Compose(
        [transforms.Resize(size=(224, 224)),
         ])

    # test set
    test_set = loader.ColorLoader(root=opts.test_path, transform=transform)
    # 복원된 사진을 보여줘야 하기 때문에 batch size 는 1
    test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    # model 정의
    net = network.Net()
    # net = net.Net().to(device)
    net.load_state_dict(torch.load('./saves/color200.ckpt'))
    net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

    # test 할 때 back propagation 이 필요 없기 때문에, with torch.no_grad(): 로 래핑을 하면, 그라디언트를 안 쓴다.
    with torch.no_grad():

        for images, colored_img in test_loader:
            # (1, 1, 224, 224)
            # images = images.to(device)
            # (1, 2, 224, 224)
            outputs = net(images)

            # tensor 를 numpy 로 변경
            images = images.numpy()
            outputs = outputs.numpy()
            colored_img = colored_img.numpy()

            # 맨처음 batch
            images = np.squeeze(images, axis=0)
            outputs = np.squeeze(outputs, axis=0)
            colored_img = np.squeeze(colored_img, axis=0)
            # print(images.shape)
            # print(outputs.shape)

            # (c, w, h) --> (w, h, c) 로 변경 (이미지 출력하기 위해서)
            images = images.transpose((1, 2, 0))
            outputs = outputs.transpose((1, 2, 0))
            colored_img = colored_img.transpose((1, 2, 0))

            # 3 channel 의 lab 이미지로 변환
            color_img = np.concatenate((images, outputs), axis=-1)
            # lab2rgb 는 0 ~ 1 사이로 나오는 것 같음. (--> 값 확인 해 보자.)
            origin_img = np.concatenate((images, colored_img), axis=-1)

            color_img = color.lab2rgb(color_img)
            origin_img = color.lab2rgb(origin_img)

            # cv2로 출력하기 위해서 format 맞춰주는 부분
            images = np.array(images).astype(np.uint8)

            color_img *= 255
            color_img = color_img[:, :, ::-1]
            color_img = np.array(color_img).astype(np.uint8)

            origin_img *= 255
            origin_img = origin_img[:, :, ::-1]
            origin_img = np.array(origin_img).astype(np.uint8)

            # resize
            images = cv2.resize(images, (448, 448))
            color_img = cv2.resize(color_img, (448, 448))
            origin_img = cv2.resize(origin_img, (448, 448))

            cv2.imshow('gray_img', images)
            cv2.imshow('color_img', color_img)
            cv2.imshow('origin_img', origin_img)
            cv2.waitKey(0)




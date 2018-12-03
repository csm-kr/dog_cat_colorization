import cv2
import numpy as np
from skimage import color


if __name__ == "__main__":

    img = cv2.imread(r"./data/test_01/cat.18.jpg", cv2.IMREAD_COLOR)
    # image_resize
    img_rgb = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

    # lab to rgb
    lab = color.rgb2lab(img_rgb)
    print("float 64", lab)

    lab = lab.astype(np.float32)
    print("float 32", lab)

    l = lab[:, :, 0][:, :, np.newaxis]
    print(l.shape)

    lab = lab.astype(np.float64)
    print("float 64", lab)

    # luminance to gray
    zero = np.zeros_like(l)
    l = np.concatenate([l, zero, zero], axis=-1)
    print(l.shape)
    l = l.astype(np.float64)
    gray = (np.clip(color.lab2rgb(l), 0, 1) * 255).astype('uint8')

    # rgb to lab
    color = (np.clip(color.lab2rgb(lab), 0, 1) * 255).astype('uint8')

    cv2.imshow("L", gray)
    cv2.imshow("input_rgb", img_rgb)
    cv2.imshow("output", color)
    cv2.waitKey(0)

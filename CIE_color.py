import cv2
import numpy as np

if __name__ == "__main__":
    img = cv2.imread(r"./data/test_01/cat.4.jpg", cv2.IMREAD_COLOR)
    gray_img = cv2.imread(r"./data/test_01/cat.4.jpg", 0)
    img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    print(img.shape)

    lab = cv2.cvtColor(src=img,
                       code=cv2.COLOR_RGB2Lab)
    l = lab[:, :, 0]
    l = l[:, :, np.newaxis]
    ab = lab[:, :, 1:]

    print("l_space : ", l.shape)
    print("ab_space : ", ab.shape)

    concatenation = np.concatenate((l, ab), axis=-1)
    print("lab_space : ", concatenation.shape)

    color = cv2.cvtColor(src=lab,
                         code=cv2.COLOR_Lab2RGB)

    cv2.imshow("concat_img", concatenation)
    cv2.imshow("lab_img", lab)
    cv2.imshow("color_img", color)

    cv2.waitKey(0)





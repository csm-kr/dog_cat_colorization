# data set 을 만드는 code 입니다. \
# 2018.09.23

import cv2
import os


def cvt_gray_scale(data_dir):

    data_name_list = os.listdir(data_dir)
    data_len = len(data_name_list)

    for i in range(data_len):
        data_path = os.path.join(data_dir, data_name_list[i])
        gray_img = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE)
        print(data_path)
        save_path = r'data/train'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_name = "g_" + data_name_list[i]
        cv2.imwrite(os.path.join(save_path, save_name), gray_img)

    cv2.imshow("gray", gray_img)
    cv2.waitKey(0)

    print("갯수는 " + str(data_len) + "개 입니다. ")
    print("done")


if __name__ == "__main__":
    data = r"C:\Users\csm81\Desktop\data\alexnet_dog_cat_img\train"
    cvt_gray_scale(data)
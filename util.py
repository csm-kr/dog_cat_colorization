import cv2
import numpy as np
import os
from skimage import color


def read_color_data_set(datadir):
    """
    x_feature 과 y_feature 을 모두 읽어드리고, 내보내는 부분
    :param datadir: 데이터셋이 있는 디렉토리
    :return: x_features : ndarray ( N, 256, 256, 1) : n - dimensional array (n 차원 array)
              y_labels   : ndarray ( N, 256, 256, 3)
    """
    # 파일이름으로 읽기
    # 파일이름이 cat 이고 뒤의 숫자로 배열한다.
    # 파일 이름 리스트

    data_list = os.listdir(datadir)
    # 데이타 갯수
    data_length = len(data_list)

    name_list = []

    # 이름 가져오는 부분
    for i in range(data_length):
        name = data_list[i].split('.')[0]
        if name == 'cat':
            name_list.append(data_list[i])

    print(name_list)
    # 정렬 하는 부분

    name_list.sort()

    cat_len = len(name_list)
    print(cat_len)

    x_ = []
    y_ = []

    for i in range(cat_len):

        img = cv2.imread(os.path.join(datadir, name_list[i]))
        img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

        Lab_img = color.rgb2lab(img)

        Lab_img = Lab_img.astype(np.float32)
        L_img = Lab_img[:, :, 0][:, :, np.newaxis]

        x_.append(L_img)
        y_.append(Lab_img)

        if np.mod(i, 1000) == 0:
            print("Loading {}/{} images...".format(i, cat_len))

    print('\nDone')

    # array to matrix
    x_data = np.array(x_)
    y_data = np.array(y_)

    print(x_data.shape)
    print(y_data.shape)

    return x_data, y_data


if __name__ == "__main__":
    data_dir = r"data/train"
    x, y, _ = read_color_data_set(data_dir)


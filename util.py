import cv2
import numpy as np
import os
import dataset as dt


def read_color_data_set(datadir):
    """
    x_feature 과 y_feature 을 모두 읽어드리고, 내보내는 부분
    :param datadir: 데이터셋이 있는 디렉토리
    :return: x_features : ndarray ( N, 227, 227, 1) : n - dimensional array (n 차원 array)
              y_labels : ndarray ( N, 227, 227, 3)
    """
    # 파일이름으로 읽기
    # 파일이름이 cat 이고 뒤의 숫자로 배열한다.
    # 파일 이름 리스트
    data_list = os.listdir(datadir)
    # 데이타 갯수
    data_length = len(data_list)

    x_list = []
    y_list = []

    # 이름 가져오는 부분
    for i in range(data_length):
        name = data_list[i].split('.')[0]
        if name == 'g_cat':
            x_list.append(data_list[i])
        elif name == 'cat':
            y_list.append(data_list[i])

    print(x_list)
    # 정렬 하는 부분

    x_list.sort()
    y_list.sort()

    cat_len = len(x_list)
    print(cat_len)

    x_features = np.empty(shape=(cat_len, 256, 256, 1), dtype=np.float32)
    y_labels = np.empty(shape=(cat_len, 256, 256, 3), dtype=np.float32)

    for i in range(cat_len):
        img_g = cv2.imread(os.path.join(datadir, y_list[i]), cv2.IMREAD_GRAYSCALE)
        img_g = cv2.resize(img_g, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        img_g = img_g[:, :, np.newaxis]
        # cv2.imshow("gray", img_g)
        x_features[i] = img_g
        img = cv2.imread(os.path.join(datadir, x_list[i]))
        img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(src=img,
                           code=cv2.COLOR_RGB2Lab)
        y_labels[i] = img

        # l = img[:, :, 0]
        # cv2.imshow("l_", l)
        # cv2.imshow("lab_color", img)
        # cv2.waitKey(0)
        if np.mod(i, 1000) == 0:
            print("Loading {}/{} images...".format(i, cat_len))

    print('\nDone')
    print(x_features.shape)
    print(y_labels.shape)

    return x_features, y_labels, x_list


if __name__ == "__main__":
    data_dir = r"data/train"
    x, y = read_color_data_set(data_dir)
    train_data = dt.DataSet(x, y)

    for epoch in range(10):

        batch_x, batch_y = train_data.next_batch(1000)
        print("epoch : ", epoch)
        print(batch_x.shape)
        print(batch_y.shape)

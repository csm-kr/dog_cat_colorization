import numpy as np


class DataSet(object):
    def __init__(self, x_features, y_labels):
        # num of data
        self._num_of_data = x_features.shape[0]
        # x features
        self._x_features = x_features
        # y features
        self._y_labels = y_labels
        # idx 들
        self._indices = np.arange(self._num_of_data, dtype=np.uint16)
        # 현재 idx
        self._now_idx = 0
        # 현재 epoch
        self._now_epoch = 0
        # 초기화
        self.reset()

    def reset(self):
        """
        몇몇의 변수를 재설정 하는 함수
        :return: 없음
        """
        self._now_epoch = 0
        self._now_idx = 0

    # getter
    @property
    def num_of_data(self):
        return self._num_of_data

    @property
    def x_features(self):
        return self._x_features

    @property
    def y_labels(self):
        return self._y_labels

    def next_batch(self, batch_size, shuffle=True):
        """
        sgd 를 이용하여 학습하기 위해서 배치 사이즈를 정하고 그만큼 잘라서 데이터를 보내주는 부분
        :param batch_size: 배치사이즈의 크기, int
        :param shuffle: 섞느냐 안섞느냐의 여부, bool
        :return: batch_x : ndarray (batch_size, 256, 256, 1)
                  batch_y : ndarray (batch_size, 256, 256, 3)
        """
        start_idx = self._now_idx

        # 처음이라면 인덱스를 섞는다.
        if self._now_idx == 0 and self._now_epoch == 0 and shuffle:
            np.random.shuffle(self._indices)

        # 넘어가는 부분
        if start_idx + batch_size > self._num_of_data:

            # epoch 증가
            self._now_epoch += 1
            rest_num = self._num_of_data - start_idx
            rest_indices = self._indices[start_idx:self._num_of_data]

            # 섞어주고
            if shuffle:
                np.random.shuffle(self._indices)

            start_idx = 0
            new_num = batch_size - rest_num
            self._now_idx = new_num
            new_indices = self._indices[start_idx:new_num]


            rest_features = self.x_features[rest_indices]
            rest_labels = self.y_labels[rest_indices]

            new_features = self.x_features[new_indices]
            new_labels = self.y_labels[new_indices]

            batch_x = np.concatenate((rest_features, new_features), axis=0)
            batch_y = np.concatenate((rest_labels, new_labels), axis=0)

        else:

            end_idx = start_idx + batch_size
            self._now_idx = end_idx

            print("end_idx : ", end_idx)

            batch_x = self.x_features[start_idx:end_idx]
            batch_y = self.y_labels[start_idx:end_idx]

        return batch_x, batch_y









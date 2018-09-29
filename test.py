import tensorflow as tf
import model
import matplotlib.pyplot as plt
import cv2
import dataset as dt
import util
import numpy as np
import os

graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(graph=graph, config=config)
model = model.Model([256, 256, 1], [256, 256, 3], batch_s=10)
saver = tf.train.Saver()
saver.restore(sess, './save/color.ckpt')  # restore learned weights

print('test started.')


test_data_dir = r"./data/test_01"
test_x, test_y, x_list = util.read_color_data_set(test_data_dir)
test_data = dt.DataSet(test_x, test_y)

print(x_list)

img = []
for i in range(10):
    temp = cv2.imread(os.path.join(test_data_dir, x_list[i]), 0)
    temp = cv2.resize(temp, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    temp = temp[:, :, np.newaxis]
    img.append(temp)

batch_size = 10
prediction_size = test_data.num_of_data
iterator = prediction_size // batch_size
_y_prediction = []

for i in range(1):

    if i == iterator:
        _batch_size = prediction_size - iterator * batch_size
    else:
        _batch_size = batch_size

    x, _ = test_data.next_batch(_batch_size)
    y_prediction = sess.run(model.logits, feed_dict={model.x: img})
    _y_prediction.append(y_prediction)

_y_prediction = np.concatenate(_y_prediction, axis=0)  # (101, num_classes)

plt.figure()

img = np.squeeze(img)
# cv2.imshow("input", img[0])

for i in range(10):

    plt.subplot(4, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img[i], cmap='gray')
    # plt.imshow(_y_prediction[i].astype(np.int32))

    plt.subplot(4, 5, i + 11)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(_y_prediction[i].astype(np.int32))

plt.show()
# 점수 내는 부분






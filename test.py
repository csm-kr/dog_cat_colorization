import tensorflow as tf
import model
import matplotlib.pyplot as plt
import cv2
import dataset as dt
import util
import numpy as np


graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(graph=graph, config=config)
model = model.Model([256, 256, 1], [256, 256, 3])
saver = tf.train.Saver()
saver.restore(sess, './save/color.ckpt')  # restore learned weights

print('test started.')

test_data = r"./data/test_01"
test_x, test_y = util.read_color_data_set(test_data)
test_data = dt.DataSet(test_x, test_y)

batch_size = 10
prediction_size = test_data.num_of_data
iterator = prediction_size // batch_size

_y_prediction = []

for i in range(iterator + 1):

    if i == iterator:
        _batch_size = prediction_size - iterator * batch_size
    else:
        _batch_size = batch_size

    x, _ = test_data.next_batch(_batch_size)
    y_prediction = sess.run(model.logits, feed_dict={model.x: test_x})

    print(y_prediction)
    print("y_prediction.shape : ", y_prediction.shape)
    # print("y_prediction.shape : ", y_prediction.shape)
    _y_prediction.append(y_prediction)

# print("_y_prediction.shape : ", _y_prediction.shape)

# 배열을 길게 늘어뜨리는 부분
# 자기 자신을 concatenate 하고 axis = 0 이면 세로 axis 가 1 이면 가로로 변경하는 것 ( 2차원일시 )
_y_prediction = np.concatenate(_y_prediction, axis=0)  # (101, num_classes)

print(_y_prediction[0])
cv2.imshow("output", _y_prediction[0])
cv2.waitKey(0)

plt.figure()
for i in range(test_data.num_of_data):
    img = test_data.x_features[i].astype(np.uint8)

    result = _y_prediction
    # print(img)
    # print(img.shape)

    print(_y_prediction.shape)

    plt.subplot(4, 5, i + 1)

    plt.xticks([])
    plt.yticks([])
    plt.imshow(_y_prediction[i])

plt.show()
# 점수 내는 부분





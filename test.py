import tensorflow as tf
import model
import cv2
import dataset as dt
import util
import numpy as np
import os
from skimage import color


graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(graph=graph, config=config)
model = model.Model([256, 256, 1], [256, 256, 3], batch_s=20)
saver = tf.train.Saver()
saver.restore(sess, './save/exp_04/color.ckpt')  # restore learned weights

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

    x, y = test_data.next_batch(_batch_size)
    y_prediction = sess.run(model.logits, feed_dict={model.x: img, model.is_train: False})

    _y_prediction.append(y_prediction)
_y_prediction = np.concatenate(_y_prediction, axis=0)  # (101, num_classes)


for num in range(batch_size):

    l = _y_prediction[num][:, :, 0][:, :, np.newaxis]
    lab = _y_prediction[num].astype(np.float64)

    # luminance to gray
    zero = np.zeros_like(l)
    l = np.concatenate([l, zero, zero], axis=-1)
    l = l.astype(np.float64)

    gray = (np.clip(color.lab2rgb(l), 0, 1) * 255).astype('uint8')
    # rgb to lab
    col = (np.clip(color.lab2rgb(lab), 0, 1) * 255).astype('uint8')

    lab_img = y[num].astype(np.float64)
    lab_img = color.lab2rgb(lab_img)
    lab_img = (np.clip(lab_img, 0, 1) * 255).astype('uint8')

    kernel = np.ones((3, 3), np.uint8)
    dilation_image = cv2.dilate(lab_img, kernel, iterations=1)  #// make dilation image
    ref = lab_img - dilation_image
    for i in range(256):
        for j in range(256):
            for c in range(3):
                if ref[i][j][c] > 200 or ref[i][j][c] < 100:
                    ref[i][j][c] = 255

    cv2.imshow("L {}".format(num + 1), gray)
    cv2.imshow("output {}".format(num + 1), col)
    #
    # cv2.imshow("original {}".format(num + 1), lab_img)

    # print(col)
    cv2.waitKey(0)








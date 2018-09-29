import tensorflow as tf
import dataset as dt
import util
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Model(object):
    def __init__(self, input_shape, output_shape, batch_s):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None] + input_shape)
        self.y = tf.placeholder(dtype=tf.float32, shape=[None] + output_shape)

        self.d, self.logits = self._build_model()
        # self.logits = self.d['logits']
        self.loss = self._build_loss()

    def _build_model(self):
        d = dict()
        # mean subtraction
        x_mean = 0.0
        x_input = self.x-x_mean
        # x_input = tf.reshape(tensor=self.x, shape=[-1, 256, 256, 1])

        # 256 * 256 * 3
        # encoder part #
        # conv1 - relu1 - pool1

        # encoder ######################################################################################################
        # input shape : (batch, 256, 256, 1)
        print("encoder start")
        print("input's shape : ", x_input.shape)

        with tf.variable_scope('layer1'):

            # conv 1
            weight = tf.get_variable(name='weight0', shape=[3, 3, 1, 64], dtype=tf.float32)
            bias = tf.get_variable(name='bias0', shape=[64], dtype=tf.float32)
            layer1 = tf.nn.conv2d(input=x_input, filter=weight, strides=[1, 1, 1, 1], padding='SAME') + bias

            # relu 1
            layer1 = tf.nn.relu(layer1)

            # batch_norma 1
            epsilon = 1e-5
            beta = tf.Variable(tf.constant(0.0, shape=[64]), trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[64]), trainable=True)
            mean, variance = tf.nn.moments(layer1, axes=[0, 1, 2])
            layer1 = tf.nn.batch_normalization(layer1, mean, variance, beta, gamma, epsilon)

            # conv 2
            weight = tf.get_variable(name='weight1', shape=[3, 3, 64, 64], dtype=tf.float32)
            bias = tf.get_variable(name='bias1', shape=[64], dtype=tf.float32)
            layer1 = tf.nn.conv2d(input=layer1, filter=weight, strides=[1, 2, 2, 1], padding='SAME') + bias

            print("layer1's shape : ", layer1.shape)
            # Conv -> (?, 128, 128, 64)

        with tf.variable_scope('layer2'):

            # conv 1
            weight = tf.get_variable(name='weight0', shape=[3, 3, 64, 128], dtype=tf.float32)
            bias = tf.get_variable(name='bias0', shape=[128], dtype=tf.float32)
            layer2 = tf.nn.conv2d(input=layer1, filter=weight, strides=[1, 1, 1, 1], padding='SAME') + bias

            # relu 1
            layer2 = tf.nn.relu(layer2)

            # batch_norma 1
            epsilon = 1e-5
            beta = tf.Variable(tf.constant(0.0, shape=[128]), trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[128]), trainable=True)
            mean, variance = tf.nn.moments(layer2, axes=[0, 1, 2])
            layer2 = tf.nn.batch_normalization(layer2, mean, variance, beta, gamma, epsilon)

            # conv 2
            weight = tf.get_variable(name='weight1', shape=[3, 3, 128, 128], dtype=tf.float32)
            bias = tf.get_variable(name='bias1', shape=[128], dtype=tf.float32)
            layer2 = tf.nn.conv2d(input=layer2, filter=weight, strides=[1, 2, 2, 1], padding='SAME') + bias

            print("layer2's shape : ", layer2.shape)
            # Conv -> (?, 64, 64, 128)

        with tf.variable_scope('layer3'):

            # conv 1
            weight = tf.get_variable(name='weight0', shape=[3, 3, 128, 256], dtype=tf.float32)
            bias = tf.get_variable(name='bias0', shape=[256], dtype=tf.float32)
            layer3 = tf.nn.conv2d(input=layer2, filter=weight, strides=[1, 1, 1, 1], padding='SAME') + bias

            # relu 1
            layer2 = tf.nn.relu(layer2)

            # batch_norma 1
            epsilon = 1e-5
            beta = tf.Variable(tf.constant(0.0, shape=[256]), trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[256]), trainable=True)
            mean, variance = tf.nn.moments(layer3, axes=[0, 1, 2])
            layer3 = tf.nn.batch_normalization(layer3, mean, variance, beta, gamma, epsilon)

            # conv 2
            weight = tf.get_variable(name='weight1', shape=[3, 3, 256, 256], dtype=tf.float32)
            bias = tf.get_variable(name='bias1', shape=[256], dtype=tf.float32)
            layer3 = tf.nn.conv2d(input=layer3, filter=weight, strides=[1, 2, 2, 1], padding='SAME') + bias

            print("layer3's shape : ", layer3.shape)
            # Conv -> (?,  32,  32, 256)

        with tf.variable_scope('layer4'):

            # conv 1
            weight = tf.get_variable(name='weight0', shape=[3, 3, 256, 512], dtype=tf.float32)
            bias = tf.get_variable(name='bias0', shape=[512], dtype=tf.float32)
            layer4 = tf.nn.conv2d(input=layer3, filter=weight, strides=[1, 1, 1, 1], padding='SAME') + bias

            # relu 1
            layer4 = tf.nn.relu(layer4)

            # batch_norma 1
            epsilon = 1e-5
            beta = tf.Variable(tf.constant(0.0, shape=[512]), trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[512]), trainable=True)
            mean, variance = tf.nn.moments(layer4, axes=[0, 1, 2])
            layer4 = tf.nn.batch_normalization(layer4, mean, variance, beta, gamma, epsilon)

            # conv 2
            weight = tf.get_variable(name='weight1', shape=[3, 3, 512, 512], dtype=tf.float32)
            bias = tf.get_variable(name='bias1', shape=[512], dtype=tf.float32)
            layer4 = tf.nn.conv2d(input=layer4, filter=weight, strides=[1, 2, 2, 1], padding='SAME') + bias

            print("layer4's shape : ", layer4.shape)
            # Conv -> (?,  16,  16, 512)

        with tf.variable_scope('layer5'):

            # conv 1
            weight = tf.get_variable(name='weight0', shape=[3, 3, 512, 512], dtype=tf.float32)
            bias = tf.get_variable(name='bias0', shape=[512], dtype=tf.float32)
            layer5 = tf.nn.conv2d(input=layer4, filter=weight, strides=[1, 1, 1, 1], padding='SAME') + bias

            # relu 1
            layer5 = tf.nn.relu(layer5)

            # batch_norma 1
            epsilon = 1e-5
            beta = tf.Variable(tf.constant(0.0, shape=[512]), trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[512]), trainable=True)
            mean, variance = tf.nn.moments(layer5, axes=[0, 1, 2])
            layer5 = tf.nn.batch_normalization(layer5, mean, variance, beta, gamma, epsilon)

            # conv 2
            weight = tf.get_variable(name='weight1', shape=[3, 3, 512, 512], dtype=tf.float32)
            bias = tf.get_variable(name='bias1', shape=[512], dtype=tf.float32)
            layer5 = tf.nn.conv2d(input=layer5, filter=weight, strides=[1, 1, 1, 1], padding='SAME') + bias

            print("layer5's shape : ", layer5.shape)
            # Conv -> (?,  16,  16, 512)

        with tf.variable_scope('layer6'):

            # conv 1
            weight = tf.get_variable(name='weight0', shape=[3, 3, 512, 512], dtype=tf.float32)
            bias = tf.get_variable(name='bias0', shape=[512], dtype=tf.float32)
            layer6 = tf.nn.conv2d(input=layer5, filter=weight, strides=[1, 1, 1, 1], padding='SAME') + bias

            # relu 1
            layer6 = tf.nn.relu(layer6)

            # batch_norma 1
            epsilon = 1e-5
            beta = tf.Variable(tf.constant(0.0, shape=[512]), trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[512]), trainable=True)
            mean, variance = tf.nn.moments(layer6, axes=[0, 1, 2])
            layer6 = tf.nn.batch_normalization(layer6, mean, variance, beta, gamma, epsilon)

            # conv 2
            weight = tf.get_variable(name='weight1', shape=[3, 3, 512, 512], dtype=tf.float32)
            bias = tf.get_variable(name='bias1', shape=[512], dtype=tf.float32)
            layer6 = tf.nn.conv2d(input=layer6, filter=weight, strides=[1, 1, 1, 1], padding='SAME') + bias

            print("layer6's shape : ", layer6.shape)
            # Conv -> (?,  16,  16, 512)

        with tf.variable_scope('layer7'):

            # conv 1
            weight = tf.get_variable(name='weight0', shape=[3, 3, 512, 512], dtype=tf.float32)
            bias = tf.get_variable(name='bias0', shape=[512], dtype=tf.float32)
            layer7 = tf.nn.conv2d(input=layer6, filter=weight, strides=[1, 1, 1, 1], padding='SAME') + bias

            # relu 1
            layer7 = tf.nn.relu(layer7)

            # batch_norma 1
            epsilon = 1e-5
            beta = tf.Variable(tf.constant(0.0, shape=[512]), trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[512]), trainable=True)
            mean, variance = tf.nn.moments(layer7, axes=[0, 1, 2])
            layer7 = tf.nn.batch_normalization(layer7, mean, variance, beta, gamma, epsilon)

            # conv 2
            weight = tf.get_variable(name='weight1', shape=[3, 3, 512, 512], dtype=tf.float32)
            bias = tf.get_variable(name='bias1', shape=[512], dtype=tf.float32)
            layer7 = tf.nn.conv2d(input=layer7, filter=weight, strides=[1, 1, 1, 1], padding='SAME') + bias

            print("layer7's shape : ", layer7.shape)
            # Conv -> (?,  16,  16, 512)

        print("decoder start")
        # decoder ######################################################################################################
        with tf.variable_scope('layer8'):
            weight = tf.get_variable(name='weight', shape=[3, 3, 256, 512], dtype=tf.float32)

            output_shape = layer7.get_shape().as_list()

            output_shape[0] = 10
            output_shape[1] *= 4
            output_shape[2] *= 4
            output_shape[3] = weight.get_shape().as_list()[2]

            layer8 = tf.nn.conv2d_transpose(value=layer7, filter=weight, output_shape=output_shape
                                            ,strides=[1, 4, 4, 1], padding='SAME')
            layer8 = tf.nn.relu(layer8)
            print("layer8's shape : ", layer8.shape)
            # DeConv -> (?, 64, 64, 256)

        with tf.variable_scope('layer9'):
            weight = tf.get_variable(name='weight', shape=[1, 1, 256, 313],  dtype=tf.float32)
            layer9 = tf.nn.conv2d(input=layer8, filter=weight, strides=[1, 1, 1, 1], padding='SAME')
            layer9 = tf.nn.relu(layer9)
            print("layer9's shape : ", layer9.shape)
            # Conv -> (?, 64, 64, 313)

        with tf.variable_scope('layer10'):
            weight = tf.get_variable(name='weight', shape=[3, 3, 2, 313], dtype=tf.float32)

            output_shape = layer9.get_shape().as_list()

            output_shape[0] = 10
            output_shape[1] *= 4
            output_shape[2] *= 4
            output_shape[3] = weight.get_shape().as_list()[2]

            layer10 = tf.nn.conv2d_transpose(value=layer9, filter=weight, output_shape=output_shape
                                             , strides=[1, 4, 4, 1], padding='SAME')
            layer10 = tf.nn.relu(layer10)
            print("layer10's shape : ", layer10.shape)

        result = tf.concat([layer10, x_input], axis=-1)
        print("result's shape : ", result.shape)
        logits = result
        d['logits'] = logits

        return d, logits

    def batch_norm_cnn(batch_image, depth):
        epsilon = 1e-5
        beta = tf.Variable(tf.constant(0.0, shape=[depth]), trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[depth]), trainable=True)
        mean, variance = tf.nn.moments(batch_image, axes=[0, 1, 2])
        norm_batch = tf.nn.batch_normalization(batch_image, mean, variance, beta, gamma, epsilon)
        return norm_batch

    def _build_loss(self):

        loss = tf.reduce_mean(tf.square(self.logits - self.y))
        loss = loss / (256 * 256)
        # 나중에 regularization 추가 해야함
        """
        로스 함수를 리턴하는 함수 loss 는 mean square error
        :return: loss
        """
        return loss


if __name__ == "__main__":

    data = r"./data/train"
    x, y, _ = util.read_color_data_set(data)
    train_data = dt.DataSet(x, y)
    print(train_data)

    # hyper parameter
    batch_size = 10
    total_epoch = 100

    m = Model(input_shape=[256, 256, 1], output_shape=[256, 256, 3], batch_s=batch_size)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(m.loss)

    # train

    graph = tf.get_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:

        save_dir = './save'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        print('Learning started. It takes sometime.')
        iterator = train_data.num_of_data // batch_size
        print("iterator : ", iterator)
        for epoch in range(total_epoch):
            avg_cost = 0
            for i in range(iterator):

                batch_x, batch_y = train_data.next_batch(batch_size)
                # print(batch_x.shape, batch_y.shape)
                _, loss = sess.run([optimizer, m.loss], feed_dict={m.x: batch_x, m.y: batch_y})
                # print("iterator : ", i, ", loss : ", loss)
                # print("batch_y = ", batch_y)
                avg_cost += loss

            print("epoch : ", epoch, ", loss : ", avg_cost/iterator)
            saver.save(sess, os.path.join(save_dir, 'color.ckpt'))  # 현재 모델의 파라미터들을 저장함

        print("Learning Done")


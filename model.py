import tensorflow as tf
import dataset as dt
import util
import os


class Model(object):
    def __init__(self, input_shape, output_shape, batch_s):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None] + input_shape)
        self.y = tf.placeholder(dtype=tf.float32, shape=[None] + output_shape)
        self.is_train = tf.placeholder(dtype=tf.bool)
        self.d, self.logits = self._build_model()
        self.loss = self._build_loss()

    def _build_model(self):
        d = dict()
        x_mean = 0.0
        x_input = self.x - x_mean
        cnt = 0

        # layer1
        with tf.variable_scope('layer{}'.format(cnt)):
            layer = tf.layers.conv2d(inputs=x_input, filters=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME",
                                     kernel_initializer=tf.initializers.random_normal(stddev=5e-2),
                                     bias_initializer=tf.constant_initializer(value=0.0),
                                     activation=tf.nn.relu)
            layer = tf.layers.conv2d(inputs=layer, filters=128, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
                                     kernel_initializer=tf.initializers.random_normal(stddev=5e-2),
                                     bias_initializer=tf.constant_initializer(value=0.0),
                                     activation=None)
            layer = tf.layers.batch_normalization(inputs=layer, training=self.is_train)
            layer = tf.nn.relu(layer)
            print("layer {} : ".format(cnt), layer.shape)
            cnt += 1

        # layer2
        with tf.variable_scope('layer{}'.format(cnt)):
            layer = tf.layers.conv2d(inputs=layer, filters=128, kernel_size=(3, 3), strides=(2, 2), padding="SAME",
                                     kernel_initializer=tf.initializers.random_normal(stddev=5e-2),
                                     bias_initializer=tf.constant_initializer(value=0.0),
                                     activation=tf.nn.relu)
            layer = tf.layers.conv2d(inputs=layer, filters=256, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
                                     kernel_initializer=tf.initializers.random_normal(stddev=5e-2),
                                     bias_initializer=tf.constant_initializer(value=0.0),
                                     activation=None)
            layer = tf.layers.batch_normalization(inputs=layer, training=self.is_train)
            layer = tf.nn.relu(layer)
            print("layer {} : ".format(cnt), layer.shape)
            cnt += 1

        # layer3
        with tf.variable_scope('layer{}'.format(cnt)):
            layer = tf.layers.conv2d(inputs=layer, filters=256, kernel_size=(3, 3), strides=(2, 2), padding="SAME",
                                     kernel_initializer=tf.initializers.random_normal(stddev=5e-2),
                                     bias_initializer=tf.constant_initializer(value=0.0),
                                     activation=tf.nn.relu)
            layer = tf.layers.conv2d(inputs=layer, filters=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
                                     kernel_initializer=tf.initializers.random_normal(stddev=5e-2),
                                     bias_initializer=tf.constant_initializer(value=0.0),
                                     activation=None)
            layer = tf.layers.batch_normalization(inputs=layer, training=self.is_train)
            layer = tf.nn.relu(layer)
            print("layer {} : ".format(cnt), layer.shape)
            cnt += 1

        # layer 4
        with tf.variable_scope('layer{}'.format(cnt)):
            layer = tf.layers.conv2d_transpose(inputs=layer,
                                               filters=256,
                                               kernel_size=(2, 2),
                                               strides=(2, 2),
                                               kernel_initializer=tf.initializers.random_normal(stddev=5e-2),
                                               bias_initializer=tf.constant_initializer(value=0.0),
                                               activation=tf.nn.relu)
            layer = tf.layers.conv2d(inputs=layer, filters=128, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
                                     kernel_initializer=tf.initializers.random_normal(stddev=5e-2),
                                     bias_initializer=tf.constant_initializer(value=0.0),
                                     activation=None)
            layer = tf.layers.batch_normalization(inputs=layer, training=self.is_train)
            layer = tf.nn.relu(layer)
            print("layer {} : ".format(cnt), layer.shape)
            cnt += 1

        # layer 5
        with tf.variable_scope('layer{}'.format(cnt)):
            layer = tf.layers.conv2d_transpose(inputs=layer,
                                               filters=128,
                                               kernel_size=(2, 2),
                                               strides=(2, 2),
                                               kernel_initializer=tf.initializers.random_normal(stddev=5e-2),
                                               bias_initializer=tf.constant_initializer(value=0.0),
                                               activation=tf.nn.relu)
            layer = tf.layers.conv2d(inputs=layer, filters=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
                                     kernel_initializer=tf.initializers.random_normal(stddev=5e-2),
                                     bias_initializer=tf.constant_initializer(value=0.0),
                                     activation=tf.nn.relu)
            layer = tf.layers.conv2d(inputs=layer, filters=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
                                     kernel_initializer=tf.initializers.random_normal(stddev=5e-2),
                                     bias_initializer=tf.constant_initializer(value=0.0),
                                     activation=None)
            layer = tf.layers.batch_normalization(inputs=layer, training=self.is_train)
            layer = tf.nn.relu(layer)
            print("layer {} : ".format(cnt), layer.shape)
            cnt += 1

        # layer 6
        with tf.variable_scope('layer{}'.format(cnt)):
            layer = tf.layers.conv2d_transpose(inputs=layer,
                                               filters=64,
                                               kernel_size=(2, 2),
                                               strides=(2, 2),
                                               kernel_initializer=tf.initializers.random_normal(stddev=5e-2),
                                               bias_initializer=tf.constant_initializer(value=0.0),
                                               activation=tf.nn.relu)
            layer = tf.layers.conv2d(inputs=layer, filters=32, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
                                     kernel_initializer=tf.initializers.random_normal(stddev=5e-2),
                                     bias_initializer=tf.constant_initializer(value=0.0),
                                     activation=tf.nn.relu)
            layer = tf.layers.conv2d(inputs=layer, filters=2, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
                                     kernel_initializer=tf.initializers.random_normal(stddev=5e-2),
                                     bias_initializer=tf.constant_initializer(value=0.0),
                                     activation=None)
            layer = tf.layers.batch_normalization(inputs=layer, training=self.is_train)
            layer = tf.nn.relu(layer)
            print("before concat layer {} : ".format(cnt), layer.shape)
            layer = tf.concat([x_input, layer], axis=-1)
            print("layer {} : ".format(cnt), layer.shape)
            d['logits'] = layer
            logits = d['logits']
            d['pred'] = layer

        return d, logits

    def _build_loss(self):

        print("logits'shape : ", self.logits.shape)
        print("y'shape : ", self.y.shape)
        loss = tf.reduce_mean(tf.square(self.logits - self.y))
        loss = tf.reduce_mean(tf.nn.l2_loss(self.logits - self.y))
        print("loss'shape : ", loss)
        return loss


if __name__ == "__main__":

    data = r"./data/train"
    x, y = util.read_color_data_set(data)
    train_data = dt.DataSet(x, y)
    print(train_data)

    # hyper parameter
    batch_size = 10
    total_epoch = 50

    m = Model(input_shape=[256, 256, 1], output_shape=[256, 256, 3], batch_s=batch_size)

    momentum = 0.9
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    update_vars = tf.trainable_variables()
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(0.0001, momentum).minimize(m.loss,
                                                                      var_list=update_vars)

    # train
    graph = tf.get_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:

        save_dir = './save/exp_07'
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
                _, loss = sess.run([optimizer, m.loss], feed_dict={m.x: batch_x, m.y: batch_y, m.is_train: True})
                print("step : ", i, " loss : ", loss)
                avg_cost += loss

            print("epoch : ", epoch, ", loss : ", avg_cost/iterator)
            # 현재 모델의 파라미터들을 저장함
            saver.save(sess, os.path.join(save_dir, 'color.ckpt'))
        print("Learning Done")


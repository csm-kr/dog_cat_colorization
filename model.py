import tensorflow as tf
import dataset as dt
import util
import os
import time


class Model(object):
    def __init__(self, input_shape, output_shape, batch_s):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None] + input_shape)
        self.y = tf.placeholder(dtype=tf.float32, shape=[None] + output_shape)
        self.is_train = tf.placeholder(dtype=tf.bool)
        self.d, self.logits = self._build_model
        self.loss = self._build_loss()

    @property
    def _build_model(self):
        d = dict()
        x_mean = 0.0
        x_input = self.x - x_mean
        cnt = 0

        # layer1
        with tf.variable_scope('layer{}'.format(cnt)):
            layer = tf.layers.conv2d(inputs=x_input, filters=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME",
                                     activation=tf.nn.relu)
            layer = tf.layers.conv2d(inputs=layer, filters=128, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
                                     activation=None)
            layer = tf.layers.batch_normalization(inputs=layer, training=self.is_train)
            layer = tf.nn.relu(layer)
            print("layer {} : ".format(cnt), layer.shape)
            cnt += 1

        # layer2
        with tf.variable_scope('layer{}'.format(cnt)):
            layer = tf.layers.conv2d(inputs=layer, filters=128, kernel_size=(3, 3), strides=(2, 2), padding="SAME",
                                     activation=tf.nn.relu)
            layer = tf.layers.conv2d(inputs=layer, filters=256, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
                                     activation=None)
            layer = tf.layers.batch_normalization(inputs=layer, training=self.is_train)
            layer = tf.nn.relu(layer)
            print("layer {} : ".format(cnt), layer.shape)
            cnt += 1

        # layer3
        with tf.variable_scope('layer{}'.format(cnt)):
            layer = tf.layers.conv2d(inputs=layer, filters=256, kernel_size=(3, 3), strides=(2, 2), padding="SAME",
                                     activation=tf.nn.relu)
            layer = tf.layers.conv2d(inputs=layer, filters=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
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
                                               activation=tf.nn.relu)
            layer = tf.layers.conv2d(inputs=layer, filters=128, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
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
                                               activation=tf.nn.relu)
            layer = tf.layers.conv2d(inputs=layer, filters=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
                                     activation=tf.nn.relu)
            layer = tf.layers.conv2d(inputs=layer, filters=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
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
                                               activation=tf.nn.relu)
            layer = tf.layers.conv2d(inputs=layer, filters=32, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
                                     activation=tf.nn.relu)
            layer = tf.layers.conv2d(inputs=layer, filters=2, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
                                     activation=None)
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
        tf.summary.scalar("loss", loss)
        return loss


if __name__ == "__main__":

    data = r"./data/train_ani"
    x, y, _ = util.read_color_data_set(data)
    train_data = dt.DataSet(x, y)

    # hyper parameter
    batch_size = 20
    total_epoch = 1000
    exp_dir = "exp_04"

    m = Model(input_shape=[256, 256, 1], output_shape=[256, 256, 3], batch_s=batch_size)

    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs/' + exp_dir)

    cur_lr = 0.01
    lr_ph = tf.placeholder(tf.float32)

    momentum = 0.9
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    update_vars = tf.trainable_variables()
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(lr_ph, momentum).minimize(m.loss,
                                                                     var_list=update_vars)

    # train
    graph = tf.get_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:

        writer.add_graph(sess.graph)
        save_dir = './save/' + exp_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        print('Learning started. It takes sometime.')
        iterator = train_data.num_of_data // batch_size
        print("iterator : ", iterator)
        min_loss = 10000

        start_time = time.time()
        for epoch in range(total_epoch):
            avg_cost = 0
            for i in range(iterator):
                batch_x, batch_y = train_data.next_batch(batch_size)
                _, loss, sum = sess.run([optimizer, m.loss, summary], feed_dict={m.x: batch_x, m.y: batch_y,
                                                                                 m.is_train: True,
                                                                                 lr_ph: cur_lr})
                if epoch != 0 and epoch % 250 == 0 and i == 0:
                    cur_lr *= 0.1
                    # decay per 20 epochs

                avg_cost += loss
                writer.add_summary(sum, global_step=i + iterator * epoch)

            print("epoch : ", epoch, ", loss : ", avg_cost/iterator, ", lr : ", cur_lr)
            # 현재 모델의 파라미터들을 저장함
            if min_loss > avg_cost/iterator:
                min_loss = avg_cost/iterator
                saver.save(sess, os.path.join(save_dir, 'color.ckpt'))
                print("save it")

        print('Total training time(sec): {}'.format(time.time() - start_time))
        print("Learning Done")


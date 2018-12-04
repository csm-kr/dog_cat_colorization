import tensorflow as tf
import util
import dataset as dt
import model as md
import optimizer as op
import evaluator as ev

data = r"./data/semi_train"
x, y, _ = util.read_color_data_set(data)
train_data = dt.DataSet(x, y)
print(train_data)

# hyper parameter
batch_size = 50
total_epoch = 500

m = md.Model(input_shape=[256, 256, 1], output_shape=[256, 256, 3], batch_s=batch_size)

# 4. make summary and writer for tensor-board
summary = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs/exp_11')

# 5. make evaluator
evaluator = ev.Evaluator()

# 6. make optimizer
optimizer = op.Optimizer(m, train_data, ev, summary)

# 7. make session options
graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# 8. session
sess = tf.Session(graph=graph, config=config)

# 9. add graph
writer.add_graph(sess.graph)

# 10. train
optimizer.train(sess, writer, save_dir='./save/exp_11')

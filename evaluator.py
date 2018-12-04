from abc import *
from sklearn.metrics import accuracy_score


class Evaluator(object):

    @property
    def worst_score(self):
        return 0.0

    @property
    def mode(self):
        return 'max'

    @abstractmethod
    def score(self, y_true, y_pred, sess=False, model=False):
        # correct_prediction = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_pred, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # # print('Accuracy:', sess.run(accuracy, feed_dict={model.x: y_true, model.x: y_pred}))
        # # return accuracy
        return accuracy_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))

    @abstractmethod
    def is_better(self, curr, best):
        score_threshold = 1e-4
        relative_eps = 1.0 + score_threshold
        return curr > best * relative_eps



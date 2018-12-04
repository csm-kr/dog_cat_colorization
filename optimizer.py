import tensorflow as tf
import os
import time
import numpy as np


class Optimizer(object):
    def __init__(self, model, train_set, evaluator, summary, val_set=None):
        self.model = model
        self.train_set = train_set
        self.val_set = val_set
        self.evaluator = evaluator
        self.summary = summary
        self.lr_placeholder = tf.placeholder(tf.float32)
        self.optimize = self._optimize_op()

        # hyper parameters
        self.init_lr = 1e-2
        self.num_epochs = 100
        self.batch_size = 50

        # reset 부분 : bad_epoch 이 계속되면 learning rate 를 낮춤
        self.num_bad_epochs = 0
        self.best_score = self.evaluator.worst_score
        self.curr_lr = self.init_lr

    def _optimize_op(self):
        """
        tf.train.Optimizer.minimize
        :return:
        """
        momentum = 0.9
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_vars = tf.trainable_variables()
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(self.lr_placeholder, momentum).minimize(self.model.loss,
                                                                                       var_list=update_vars)
        return optimizer

    def _step(self, sess):
        """
        training 을 위해서 update 딱 한번 하는 부분 여기서 summary 도 업데이트 해야함
        :param sess:
        :return:
        """
        x, y = self.train_set.next_batch(batch_size=self.batch_size, shuffle=True)
        _, loss, pred, summ = sess.run([self.optimize, self.model.loss, self.model.prediction, self.summary],
                                       feed_dict={self.model.x: x, self.model.y: y, self.model.is_train: True,
                                                  self.lr_placeholder: self.curr_lr})

        # return 에 y 가 들어가는 이유는 val 을 위해서
        return loss, y, pred, summ

    def _update_learning_rate(self):
        """
        learning rate 를 줄이는 함수 하한선 존재 eps
        """
        lr_patience = 10
        lr_decay = 0.1
        eps = 1e-7

        if self.num_bad_epochs > lr_patience:
            # bae epoch 이 10번을 넘으면 0.1 곱한 값을 새로운 rate 로 업데이트
            new_lr = self.curr_lr * lr_decay
            # 하한선을 두는 것
            if self.curr_lr - new_lr > eps:
                self.curr_lr = new_lr
            self.num_bad_epochs = 0

    def train(self, sess, writer, save_dir):
        """
        model 을 optimize 하기 위한 모델
        :param sess: 세션
        :param writer: tensorboard 를 위한 writer
        :param save_dir: 어디다 저장하니
        """
        saver = tf.train.Saver()  # 저장하기 위해서
        sess.run(tf.global_variables_initializer())  # 전체 파라미터들을 초기화함

        print('Learning started. It takes sometime.')
        iterator = self.train_set.num_of_data // self.batch_size
        print("Number of a iteration : ", iterator)
        print("Number of whole iterations : ", iterator * self.num_epochs)

        # 각 스텝에 learning 한 것들 넣기 위한 부분
        step_losses, step_scores, eval_scores = [], [], []

        # 시간재기
        start_time = time.time()
        for epoch in range(self.num_epochs):
            for i in range(iterator):
                step_loss, step_y_true, step_y_pred, summary = self._step(sess)
                step_losses.append(step_loss)

                # writer 저장
                writer.add_summary(summary=summary, global_step=epoch * iterator + i)

                # epoch 의 마지막 단계에서
                if i == iterator-1:
                    step_score = self.evaluator.score(step_y_true, step_y_pred)
                    step_scores.append(step_score)

                    # validation score --> 있으면, is_better 의 기준이 val 만이 된다.
                    if self.val_set is not None:

                        eval_y_pred = self.model.predict(sess, self.val_set)
                        eval_score = self.evaluator.score(self.val_set.y_data, eval_y_pred)
                        eval_scores.append(eval_score)

                        # eval_score 가 있다면, 최고의 eval_score 이 curr_score 가 된다.
                        print('[epoch {}]\tloss: {:.6f} |Train score: {:.6f} |Eval score: {:.6f} |lr: {:.6f}' \
                              .format(epoch + 1, np.mean(step_loss), step_score, eval_score, self.curr_lr))
                        curr_score = eval_score

                    else:
                        # eval_score 가 없다면, 최고의 step_score 이 curr_score 가 된다.
                        print('[epoch {}]\tloss: {} |Train score: {:.6f} |lr: {:.6f}' \
                              .format(epoch + 1, np.mean(step_loss), step_score, self.curr_lr))
                        curr_score = step_score

                    # compare scores
                    if self.evaluator.is_better(curr_score, self.best_score):
                        self.best_score = curr_score
                        self.num_bad_epochs = 0
                        saver.save(sess, os.path.join(save_dir, 'model.ckpt'))  # 현재 모델의 파라미터들을 저장함
                    else:
                        self.num_bad_epochs += 1

                    # learning rate update
                    self._update_learning_rate()

        # 시간재기
        print('Total training time(sec): {}'.format(time.time() - start_time))
        print("Learning Done")

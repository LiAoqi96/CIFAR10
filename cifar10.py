import ops
import tensorflow as tf
import cifar10_input
import os
import time
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Cifar10(ops.BasicOps):
    def __init__(self, sess, data_dir, batch_size, training, checkpoint, log_dir,
                 mode='cnn', weight_decay=None, data_format='channels_last'):
        super().__init__(training, data_format)

        self.batch_size = batch_size
        self.sess = sess
        self.mode = mode
        self.checkpoint = os.path.join(checkpoint, mode)
        self.log_dir = log_dir
        self.data_dir = data_dir

        self.dataset = cifar10_input.Cifar10DataSet(data_dir, 'train')
        self.iterator = self.dataset.get_iterator(batch_size)
        self.images = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='images')
        self.labels = tf.placeholder(tf.int64, shape=[None], name='labels')

        self.keep_prop = tf.placeholder(tf.float32, name='keep_prop')
        self.output = self.inference(self.images)

        self.loss = tf.losses.sparse_softmax_cross_entropy(
            logits=self.output, labels=self.labels)
        self.loss = tf.reduce_mean(self.loss)
        if weight_decay:
            model_params = tf.trainable_variables()
            self.loss += weight_decay * tf.add_n(
                [tf.nn.l2_loss(v) for v in model_params])

        correct_prediction = tf.nn.in_top_k(self.output, self.labels, 1)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        self.precision = tf.Variable(0, False, name='precision')
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('precision', self.precision)

        self.saver = tf.train.Saver()
        self.global_step = tf.train.get_or_create_global_step()

    def inference(self, images):
        if self._data_format == 'channels_first':
            # Computation requires channels_first.
            images = tf.transpose(images, [0, 3, 1, 2])

        images = images / 128 - 1
        if self.mode == 'res_net':
            x = self.relu(
                self.batch_norm(self.conv2d(images, 16)))
            for i in range(1, 8):
                x = self.res_block_v2(x, 16)
            for i in range(1, 8):
                x = self.res_block_v2(x, 32, s=2 if i == 1 else 1)
            for i in range(1, 8):
                x = self.res_block_v2(x, 64, s=2 if i == 1 else 1)
            x = self.relu(self.batch_norm(x))
            x = self.global_avg_pool(x)
            output = self.dense(x, 10, name='output')
        elif self.mode == 'dense_net':
            x = self.relu(
                self.batch_norm(self.conv2d(images, 16)))
            x = self.dense_block(x, 12, 12)
            x = self.transition(x, 0.5)
            x = self.dense_block(x, 12, 12)
            x = self.transition(x, 0.5)
            x = self.dense_block(x, 12, 12)
            x = self.relu(self.batch_norm(x))
            x = self.global_avg_pool(x)
            output = self.dense(x, 10, name='output')
        elif self.mode == 'cnn':
            h_conv1 = self.relu(
                self.batch_norm(self.conv2d(images, 64)))
            h_pool1 = self.max_pool(h_conv1)
            h_conv2 = self.relu(self.batch_norm(self.conv2d(h_pool1, 128)))
            h_pool2 = self.max_pool(h_conv2)
            h_conv3 = self.relu(self.batch_norm(self.conv2d(h_pool2, 256)))
            h_pool3 = self.max_pool(h_conv3)
            h_dense1 = self.relu(self.dense(tf.layers.flatten(h_pool3), 1024, keep_prop=self.keep_prop))
            output = self.dense(h_dense1, 10, name='output')
        else:
            print('Choose properly mode')
            return
        return output

    def train(self, steps=100000):
        global_step = self.global_step
        lr = tf.train.piecewise_constant(global_step, [10000, 50000], [0.001, 0.0001, 0.00001])
        train_op = tf.train.AdamOptimizer(lr).minimize(self.loss, global_step=global_step)

        tf.summary.scalar('learning_rate', lr)
        tf.global_variables_initializer().run()
        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        if self.load(self.sess):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        next_element = self.iterator.get_next()
        t = time.time()
        for step in range(steps):
            images, labels = self.sess.run(next_element)
            _, loss = self.sess.run([train_op, self.loss],
                                    feed_dict={
                                        self.images: images,
                                        self.labels: labels,
                                        self.keep_prop: 0.5
                                    })

            if step % 100 == 0:
                accuracy = self.sess.run(self.accuracy,
                                         feed_dict={
                                             self.images: images,
                                             self.labels: labels,
                                             self.keep_prop: 1.0
                                         })
                print('step %d: loss %.5f,accuracy %.3f,spend %.3fs' % (step, loss, accuracy, time.time() - t))
                t = time.time()
                summary_str = self.sess.run(summary,
                                            feed_dict={
                                                self.images: images,
                                                self.labels: labels,
                                                self.keep_prop: 1.0
                                            })
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
            if (step + 1) % 1000 == 0:
                self.val(step)
                checkpoint_file = os.path.join(self.checkpoint, 'model.ckpt')
                self.saver.save(self.sess, checkpoint_file, step)

        self.sess.close()

    def val(self, step, subset='validation'):
        self._is_training = False
        iterator = cifar10_input.Cifar10DataSet(self.data_dir, subset).get_iterator(100)
        next_element = iterator.get_next()
        accuracy = 0
        for i in range(100):
            images, labels = self.sess.run(next_element)
            accuracy += self.sess.run(self.accuracy,
                                      feed_dict={
                                          self.images: images,
                                          self.labels: labels,
                                          self.keep_prop: 1.0
                                      })
        precision = accuracy / 100
        print('Step: %d precision @ 1 = %.3f' % (step, precision))
        self.sess.run(tf.assign(self.precision, precision))
        self._is_training = True

    def eval(self, subset='eval'):
        if self.load(self.sess):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            return
        step = self.sess.run(self.global_step)

        iterator = cifar10_input.Cifar10DataSet(self.data_dir, subset).get_iterator(100)
        next_element = iterator.get_next()
        accuracy = 0
        for i in range(100):
            images, labels = self.sess.run(next_element)
            accuracy += self.sess.run(self.accuracy,
                                      feed_dict={
                                          self.images: images,
                                          self.labels: labels,
                                          self.keep_prop: 1.0
                                      })
        precision = accuracy / 100
        print('Step: %d precision @ 1 = %.3f' % (step, precision))

    def load(self, sess):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = self.checkpoint

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            step = ckpt_name.split('-')[1]
            sess.run(tf.assign(self.global_step, np.int32(step)))
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format('model_' + str(step)))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False

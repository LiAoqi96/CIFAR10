import tensorflow as tf
from cifar10 import Cifar10

flags = tf.flags
flags.DEFINE_string('data_dir', './data', 'Directory of data')
flags.DEFINE_string('log_dir', './logs', 'Directory of logs')
flags.DEFINE_string('checkpoint', './checkpoint', 'Directory to save model')
flags.DEFINE_string('mode', 'cnn', 'Network mode.[cnn, res_net, dense_net]')
flags.DEFINE_integer('steps', 100000, 'Steps to train.')
flags.DEFINE_boolean('train', True, 'Train or eval')
FLAGS = flags.FLAGS


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.checkpoint)

    with tf.Graph().as_default():
        run_config = tf.ConfigProto(allow_soft_placement=True)
        run_config.gpu_options.allow_growth = True
        with tf.Session(config=run_config) as sess:
            model = Cifar10(sess=sess, data_dir=FLAGS.data_dir, batch_size=128,
                            training=FLAGS.train, checkpoint=FLAGS.checkpoint, log_dir=FLAGS.log_dir,
                            mode=FLAGS.mode, weight_decay=1e-4)
            if FLAGS.train:
                model.train(steps=FLAGS.steps)
            else:
                model.eval('eval')


if __name__ == '__main__':
    tf.app.run()

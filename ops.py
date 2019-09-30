import tensorflow as tf


class BasicOps(object):
    def __init__(self, is_training, data_format, batch_norm_decay=0.997, batch_norm_epsilon=1e-5):
        """ResNet constructor.

        Args:
          is_training: if build training or inference model.
          data_format: the data_format used during computation.
                       one of 'channels_first' or 'channels_last'.
        """
        self._batch_norm_decay = batch_norm_decay
        self._batch_norm_epsilon = batch_norm_epsilon
        self._is_training = is_training
        assert data_format in ('channels_first', 'channels_last')
        self._data_format = data_format

    def batch_norm(self, x):
        if self._data_format == 'channels_first':
            data_format = 'NCHW'
        else:
            data_format = 'NHWC'
        return tf.contrib.layers.batch_norm(x,
                                            decay=self._batch_norm_decay,
                                            epsilon=self._batch_norm_epsilon,
                                            center=True,
                                            scale=True,
                                            updates_collections=None,
                                            is_training=self._is_training,
                                            fused=True,
                                            data_format=data_format)

    def conv2d(self, input_, out_dim, k=3, s=1, use_bias=False, name='conv'):
        with tf.name_scope(name):
            return tf.layers.conv2d(input_,
                                    filters=out_dim,
                                    kernel_size=k,
                                    strides=s,
                                    padding='same',
                                    use_bias=use_bias,
                                    data_format=self._data_format)

    def relu(self, x):
        return tf.nn.relu(x)

    def res_block_v1(self, input_, out_dim, k=3, s=1, name='res_block_v1'):
        with tf.name_scope(name):
            orig_x = input_

            x = self.conv2d(input_, out_dim, k, s)
            x = self.batch_norm(x)
            x = self.relu(x)

            x = self.conv2d(x, out_dim, k)
            x = self.batch_norm(x)

            if s != 1:
                orig_x = self.avg_pool(orig_x, s, s)
                pad = out_dim // 4
                if self._data_format == 'channels_first':
                    orig_x = tf.pad(orig_x, [[0, 0], [pad, pad], [0, 0], [0, 0]])
                else:
                    orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0], [pad, pad]])
            x = self.relu(tf.add(x, orig_x))
            return x

    def res_block_v2(self, input_, out_dim, k=3, s=1, name='res_block_v2'):
        with tf.name_scope(name):
            orig_x = input_

            x = self.batch_norm(input_)
            x = self.relu(x)

            x = self.conv2d(x, out_dim, k, s)
            x = self.batch_norm(x)
            x = self.relu(x)

            x = self.conv2d(x, out_dim, k)

            if s != 1:
                orig_x = self.avg_pool(orig_x, s, s)
                pad = out_dim // 4
                if self._data_format == 'channels_first':
                    orig_x = tf.pad(orig_x, [[0, 0], [pad, pad], [0, 0], [0, 0]])
                else:
                    orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0], [pad, pad]])
            x = tf.add(x, orig_x)
        return x

    def bottle_res_block(self, input_, out_dim, k, s, name='bottle_res_blok'):
        with tf.name_scope(name):
            orig_x = input_

            x = self.batch_norm(input_)
            x = self.relu(x)

            x = self.conv2d(x, out_dim // 4, 1, s)
            x = self.batch_norm(x)
            x = self.relu(x)

            x = self.conv2d(x, out_dim // 4, k)
            x = self.batch_norm(x)
            x = self.relu(x)

            x = self.conv2d(x, out_dim, 1)
            if s != 1:
                orig_x = self.conv2d(orig_x, out_dim, 1, s)
            x = tf.add(x, orig_x)
        return x

    def dense_layer(self, input_, out_dim=12, bc_mode=False, name='dense_layer'):
        with tf.name_scope(name):
            if bc_mode:
                output = self.relu(self.batch_norm(input_))
                output = self.conv2d(output, out_dim * 4, k=1)
                output = self.relu(self.batch_norm(output))
            else:
                output = self.relu(self.batch_norm(input_))
            output = self.conv2d(output, out_dim)
        return output

    def dense_block(self, input_, layers=12, out_dim=12, bc_mode=False, name='dense_block'):
        with tf.name_scope(name):
            for i in range(layers):
                output = self.dense_layer(input_, out_dim, bc_mode)
                input_ = tf.concat([input_, output], axis=1 if self._data_format == 'channels_first' else 3)
        return input_

    def transition(self, input_, reduction=0.5, name='transition'):
        in_dim = input_.shape[1] if self._data_format == 'channels_first' else input_.shape[3]
        with tf.name_scope(name):
            output = self.relu(self.batch_norm(input_))
            output = self.conv2d(output, int(in_dim * reduction), k=1)
            output = self.avg_pool(output)
        return output

    def max_pool(self, input_, p=2, s=2, name='max_pool'):
        with tf.name_scope(name):
            pool = tf.layers.max_pooling2d(input_, p, s, padding='same', data_format=self._data_format)
        return pool

    def avg_pool(self, input_, p=2, s=2, name='avg_pool'):
        with tf.name_scope(name):
            pool = tf.layers.average_pooling2d(input_, p, s, padding='same', data_format=self._data_format)
        return pool

    def global_avg_pool(self, input_, name='global_avg_pool'):
        with tf.name_scope(name):
            if self._data_format == 'channels_first':
                x = tf.reduce_mean(input_, [2, 3])
            else:
                x = tf.reduce_mean(input_, [1, 2])
        return x

    def dense(self, input_, units, keep_prop=1.0, name='dense'):
        with tf.name_scope(name):
            out = tf.nn.dropout(tf.layers.dense(input_, units), keep_prop)
        return out

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import tensorflow as tf

IMAGE_SIZE = 32
IMAGE_DEPTH = 3


class Cifar10DataSet(object):
    def __init__(self, data_dir, subset='train', use_distortion=True):
        self.data_dir = data_dir
        self.subset = subset
        self.use_distortion = use_distortion

    def get_filename(self):
        return os.path.join(self.data_dir, self.subset + '.tfrecords')

    def parser(self, serialized_example):
        feature = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)
            })
        image = tf.decode_raw(feature['image'], tf.uint8)
        image.set_shape([IMAGE_SIZE * IMAGE_SIZE * IMAGE_DEPTH])

        image = tf.cast(tf.transpose(tf.reshape(image, [IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE]), [1, 2, 0]), tf.float32)
        label = tf.cast(feature['label'], tf.int32)

        image = self.preprocess(image)
        return image, label

    def get_iterator(self, batch_size):
        filename = self.get_filename()

        dataset = tf.data.TFRecordDataset(filename).repeat()
        dataset = dataset.map(self.parser, num_parallel_calls=batch_size)

        if self.subset == 'train':
            min_queue_examples = int(
                Cifar10DataSet.num_examples_per_epoch(self.subset) * 0.4)
            # Ensure that the capacity is sufficiently large to provide good random
            # shuffling.
            dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)

        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator

    def preprocess(self, image):
        if self.subset == 'train' and self.use_distortion:
            image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
            image = tf.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH])
            image = tf.image.random_flip_left_right(image)
        return image

    @staticmethod
    def num_examples_per_epoch(subset='train'):
        if subset == 'train':
            return 45000
        elif subset == 'validation' or 'eval':
            return 10000
        else:
            raise ValueError('Invalid data subset "%s"' % subset)

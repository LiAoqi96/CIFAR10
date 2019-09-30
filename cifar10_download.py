import requests
import tensorflow as tf
import os
import pickle
import tarfile

CIFAR_FILENAME = 'cifar-10-python.tar.gz'
CIFAR_DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/' + CIFAR_FILENAME
CIFAR_LOCAL_FOLDER = 'cifar-10-batches-py'


def download_and_extract(data_dir):
    path = os.path.join(data_dir, CIFAR_FILENAME)
    if not os.path.exists(path):
        r = requests.get(CIFAR_DOWNLOAD_URL)
        with open(path, 'wb') as code:
            code.write(r.content)
        tarfile.open(path, 'r:gz').extractall(data_dir)


def get_file_names():
    file_names = dict()
    file_names['train'] = ['data_batch_%d' % i for i in range(1, 5)]
    file_names['validation'] = ['data_batch_5']
    file_names['eval'] = ['test_batch']
    return file_names


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecord(input_files, output_file):
    print('Generating %s' % output_file)
    with tf.python_io.TFRecordWriter(output_file) as record_writer:
        for input_file in input_files:
            with tf.gfile.Open(input_file, 'rb') as f:
                data_dict = pickle.load(f, encoding='bytes')
            data = data_dict[b'data']
            labels = data_dict[b'labels']

            for i in range(len(labels)):
                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image': _bytes_feature(data[i].tobytes()),
                        'label': _int64_feature(labels[i])
                    }))

                record_writer.write(example.SerializeToString())


def main(data_dir):
    print('Download from {} and extract.'.format(CIFAR_DOWNLOAD_URL))
    download_and_extract(data_dir)
    file_names = get_file_names()
    input_dir = os.path.join(data_dir, CIFAR_LOCAL_FOLDER)

    for mode, files in file_names.items():
        input_files = [os.path.join(input_dir, f) for f in files]
        output_file = os.path.join(data_dir, mode + '.tfrecords')

        try:
            os.remove(output_file)
        except IOError:
            pass
        convert_to_tfrecord(input_files, output_file)
    print('Done!')


if __name__ == '__main__':
    flags = tf.flags
    flags.DEFINE_string('data_dir', default='./data', help='Directory to save data')
    FLAGS = flags.FLAGS

    main(FLAGS.data_dir)

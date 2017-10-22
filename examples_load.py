from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import tensorflow as tf

import tfrecord


def load_classification():
    print('Load tfrecord for classification!')
    TR = tfrecord.TfrecordData(
        tfrecord_path='classification_tfrecord',
        batch_size=5,
        shuffle=True,
        num_threads=4)  # num_threads should be 1 for not shuffle
    fields = TR.fields()
    print('fields:', fields)

    for i in range(2):
        img_batch, class_batch = TR.batch(['img', 'class'])
        for img, clc in zip(img_batch, class_batch):
            plt.imshow(img)
            plt.title('class: %d' % clc)
            plt.show()


def load_regression():
    print('Load tfrecord for regression!')
    TR = tfrecord.TfrecordData(
        tfrecord_path='regression_tfrecord',
        batch_size=5,
        shuffle=False,
        num_threads=1)  # num_threads should be 1 for not shuffle
    fields = TR.fields()
    print('fields:', fields)

    for i in range(2):
        img_batch, targets_batch = TR.batch(['img', 'targets'])
        for img, targets in zip(img_batch, targets_batch):
            plt.imshow(img)
            print('targets:', targets)
            plt.show()


def load_multi_label():
    # preprocess
    def img_preprecess_fn(img):
        img = tf.image.resize_images(img, [256, 128])
        img = img / 255
        return img

    print('Load tfrecord for multiple label task!')
    TR = tfrecord.TfrecordData(
        tfrecord_path='multi_label_tfrecord',
        batch_size=5,
        shuffle=False,
        num_threads=1,
        preprocess_fns={
            'img': img_preprecess_fn
        })  # num_threads should be 1 for not shuffle
    fields = TR.fields()
    print('fields:', fields)

    for i in range(2):
        img_batch, class_batch, targets_batch = TR.batch(['img', 'class', 'targets'])
        for img, clc, targets in zip(img_batch, class_batch, targets_batch):
            plt.imshow(img)
            plt.title('class: %d' % clc)
            print('targets:', targets)
            plt.show()


if __name__ == '__main__':
    load_classification()
    load_regression()
    load_multi_label()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from glob import glob
import io
import os

import numpy as np
from PIL import Image

import tfrecord

img_list = glob(os.path.join('images', '*.jpg'))

imgs = []
for img_path in img_list:
    img = Image.open(img_path)
    img = img.resize((128, 128))
    imgs.append(img)

# set classification labels - id
classification_labels = range(len(imgs))

# set regression labels - 10 dimension label
regression_labels = list(np.random.rand(len(imgs), 10))


def classification():
    print('Create tfrecord for classification!')

    # create a writer
    writer = tfrecord.ImageLablePairTfrecordCreator(
        save_dir='classification_tfrecord',
        label_type='classification',
        encode_type=None,
        data_name='img',
        label_name='class',
        compression_type=0)

    # dump data and label
    for img, label in zip(imgs, classification_labels):
        writer.add(np.array(img), np.array(label))

    writer.close()


def regression():
    print('Create tfrecord for regression!')

    # create a writer
    writer = tfrecord.ImageLablePairTfrecordCreator(
        save_dir='regression_tfrecord',
        label_type='regression',
        encode_type='jpg',
        quality=80,
        data_name='img',
        label_name='targets',
        compression_type=1)

    # dump data and label
    for img, label in zip(imgs, regression_labels):
        writer.add(np.array(img), np.array(label).astype(np.float32))

    writer.close()


def multi_label():
    print('Create tfrecord for multiple label task!')

    # create a writer
    writer = tfrecord.BytesTfrecordCreator(
        save_dir='multi_label_tfrecord',
        compression_type=2)

    # dump data and label
    # Different from the easy usage of tfrecord.ImageLablePairTfrecordCreator,
    # you should convert the data and labels to bytes by yourself and add informations
    for img, clc_label, reg_label in zip(imgs, classification_labels, regression_labels):
        # encode image and convert get bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='png')
        img_bytes = img_bytes.getvalue()

        # convert labels to bytes
        clc_label_bytes = np.array(clc_label).astype(np.int64).tobytes()
        reg_label_bytes = np.array(reg_label).astype(np.float32).tobytes()

        # add an item
        writer.add({
            'img': img_bytes,
            'class': clc_label_bytes,
            'targets': reg_label_bytes
        })

    # add info
    writer.add_info('img', 'png', [128, 128, 3])
    writer.add_info('class', 'int64', [])
    writer.add_info('targets', 'float32', [10])
    writer.close()


if __name__ == '__main__':
    classification()
    regression()
    multi_label()

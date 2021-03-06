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
        save_path='classification_tfrecord',
        encode_type=None,
        data_name='img',
        compression_type=0)

    # dump data and label
    for img, label in zip(imgs, classification_labels):
        writer.add(np.array(img), {"class": np.array(label)})

    writer.close()


def regression():
    print('Create tfrecord for regression!')

    # create a writer
    writer = tfrecord.ImageLablePairTfrecordCreator(
        save_path='regression_tfrecord',
        encode_type='jpg',
        quality=80,
        data_name='img',
        block_size=2,   # each tfrecord file contains `block_size` items. If `block_size` is None, there is one tfrecord file containing all data.
        compression_type=1)

    # dump data and label
    for img, label in zip(imgs, regression_labels):
        writer.add(np.array(img), {"targets": np.array(label).astype(np.float32)})

    writer.close()


def multiple_label():
    print('Create tfrecord for multiple label task!')

    # create a writer
    writer = tfrecord.ImageLablePairTfrecordCreator(
        save_path='multiple_label_tfrecord',
        encode_type='jpg',
        quality=80,
        data_name='img',
        compression_type=1)

    # dump data and label
    for img, clc_label, reg_label in zip(imgs, classification_labels, regression_labels):
        writer.add(np.array(img), {"targets": np.array(reg_label).astype(np.float32),
                                   "class": np.array(clc_label)})

    writer.close()


def arbitrary_data():
    print('Create tfrecord for arbitrary data!')

    # create a writer
    writer = tfrecord.BytesTfrecordCreator(
        save_path='arbitrary_data_tfrecord',
        compression_type=2)

    # dump arbitrary data
    # Different from the easy usage of tfrecord.ImageLablePairTfrecordCreator,
    # you should convert the data to bytes by yourself and add informations
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
    multiple_label()
    arbitrary_data()

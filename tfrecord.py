from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import json
import os

import numpy as np
from PIL import Image
import tensorflow as tf

__metaclass__ = type


DECODERS = {
    'png': {'decoder': tf.image.decode_png, 'decode_param': dict()},
    'jpg': {'decoder': tf.image.decode_jpeg, 'decode_param': dict()},
    'jpeg': {'decoder': tf.image.decode_jpeg, 'decode_param': dict()},
    'uint8': {'decoder': tf.decode_raw, 'decode_param': dict(out_type=tf.uint8)},
    'float32': {'decoder': tf.decode_raw, 'decode_param': dict(out_type=tf.float32)},
    'int64': {'decoder': tf.decode_raw, 'decode_param': dict(out_type=tf.int64)},
}
ALLOWED_TYPES = DECODERS.keys()


class BytesTfrecordCreator(object):
    """BytesTfrecordCreator.

    `compression_type`:
        0: NONE
        1: ZLIB
        2: GZIP
    """

    def __init__(self, save_dir, compression_type=0):
        options = tf.python_io.TFRecordOptions(compression_type)
        self.writer = tf.python_io.TFRecordWriter(
            os.path.join(save_dir, 'data.tfrecord'), options)
        self.info_f = open(os.path.join(save_dir, 'info.txt'), 'w')

        self.feature_names = None
        self.info_names = []  # is the same as self.feature_names except for item order
        self.info_list = []

        self.compression_type = compression_type

        self.closed = False

    @staticmethod
    def bytes_feature(values):
        """Return a TF-Feature of bytes.

        Args:
          values: A byte string or list of byte strings.

        Returns:
          a TF-Feature.
        """
        if not isinstance(values, (tuple, list)):
            values = [values]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

    @staticmethod
    def bytes_tfexample(bytes_dict):
        """Convert bytes to tfexample.

        `bytes_dict` example:
            bytes_dict = {
                'img': img_bytes,
                'id': id_bytes,
                'attr': attr_bytes,
                'point': point_bytes
            }
        """
        feature_dict = {}
        for key, value in bytes_dict.items():
            feature_dict[key] = BytesTfrecordCreator.bytes_feature(value)
        return tf.train.Example(features=tf.train.Features(feature=feature_dict))

    def add(self, feature_bytes_dict):
        """Add example.

        `feature_bytes_dict` example:
            feature_bytes_dict = {
                'img': img_bytes,
                'id': id_bytes,
                'attr': attr_bytes,
                'point': point_bytes
            }
        """
        if self.feature_names is None:
            self.feature_names = feature_bytes_dict.keys()
        else:
            assert self.feature_names == feature_bytes_dict.keys(), \
                'Feature names are inconsistent!'

        tfexample = BytesTfrecordCreator.bytes_tfexample(feature_bytes_dict)
        self.writer.write(tfexample.SerializeToString())

    def add_info(self, name, dtype_or_format, shape):
        """Add feature informations.

        example:
            add_info('img', 'png', [64, 64, 3])
        """
        assert name not in self.info_names, 'info name duplicated!'

        dtype_or_format = dtype_or_format.lower()
        assert dtype_or_format in ALLOWED_TYPES, \
            "`dtype_or_format` should be in the list of %s!" \
            % str(ALLOWED_TYPES)

        self.info_names.append(name)
        self.info_list.append(dict(name=name,
                                   dtype_or_format=dtype_or_format,
                                   shape=shape))

    def close(self):
        assert sorted(self.feature_names) == sorted(self.info_names), \
            "Feature informations should be added by function 'add_info(...)!'"

        # save info
        self.info_list.append({'compression_type': self.compression_type})
        info_str = json.dumps(self.info_list)
        info_str = info_str.replace('}, {', '},\n {')
        self.info_f.write(info_str)

        # close files
        self.writer.close()
        self.info_f.close()

        self.closed = True

    def __del__(self):
        if not self.closed:
            self.close()


class DataLablePairTfrecordCreator(BytesTfrecordCreator):
    """DataLablePairTfrecordCreator.

    `label_type`:
        'classification' -> scalar int64 label
        'regression'     -> scalar or vector float32 label
    `label_shape`: if `label_type` is 'regression', then `label_shape` should be given

    `compression_type`:
        0: NONE
        1: ZLIB
        2: GZIP
    """

    def __init__(self, save_dir, data_shape, data_dtype_or_format, label_type,
                 label_shape=None, data_name='data', label_name='label', compression_type=0):
        super(DataLablePairTfrecordCreator, self).__init__(save_dir, compression_type)

        assert label_type in ['classification', 'regression'], \
            "`label_type` should be 'classification' or 'regression'!"
        if label_type is 'regression' and label_shape is None:
            raise Exception('Regression: `label_shape` should be given!')

        self.data_shape = data_shape
        self.data_dtype_or_format = data_dtype_or_format
        self.data_name = data_name
        self.label_type = label_type
        self.label_shape = label_shape
        self.label_name = label_name

    def add(self, data, label):
        """Add example.

        `data` and `label` can already be byte string or numpy array,
        the function will convert them to byte string anyway

        Note: no shape of dtype check if byte string is given, which is unsave
        """
        assert isinstance(data, (str, np.ndarray)) and \
            isinstance(label, (str, np.ndarray)), \
            '`data` and `label` should be byte string or numpy array!'
        if isinstance(data, np.ndarray):
            assert data.dtype.name == self.data_dtype_or_format, \
                'dtype of `data` should be %s!' % self.data_dtype_or_format
            assert data.shape == tuple(self.data_shape), \
                'shape of `data` should be %s!' % str(tuple(self.data_shape))
            data = data.tobytes()
        if isinstance(label, np.ndarray):
            if self.label_type == 'classification':
                assert label.dtype.name == 'int64', \
                    'Classification: dtype of `label` should be int64!'
                assert label.shape == (), \
                    'Classification: `label` should be a scalar!'
            elif self.label_type == 'regression':
                assert label.dtype.name == 'float32', \
                    'Regression: dtype of `label` should be float32!'
                assert label.shape == tuple(self.label_shape), \
                    'Regression: shape of `label` should be %s!' \
                    % str(tuple(self.label_shape))
            label = label.tobytes()

        super(DataLablePairTfrecordCreator, self).add({self.data_name: data,
                                                       self.label_name: label})

    def close(self):
        self.add_info(self.data_name, self.data_dtype_or_format, self.data_shape)
        if self.label_type == 'classification':
            self.add_info(self.label_name, 'int64', ())
        elif self.label_type == 'regression':
            self.add_info(self.label_name, 'float32', self.label_shape)

        super(DataLablePairTfrecordCreator, self).close()


class ImageLablePairTfrecordCreator(DataLablePairTfrecordCreator):
    """ImageLablePairTfrecordCreator.

    `label_type`:
        'classification' -> scalar int64 label
        'regression'     -> scalar or vector float32 label

    `encode_type`: in [None, 'png', 'jpg', 'jpeg']
    `quality`: for 'jpg' or 'jpeg'

    `compression_type`:
        0: NONE
        1: ZLIB
        2: GZIP
    """

    def __init__(self, save_dir, label_type, encode_type, quality=95,
                 data_name='data', label_name='label', compression_type=0):
        super(ImageLablePairTfrecordCreator, self).__init__(
            save_dir, None, None, label_type, (), data_name, label_name, compression_type)

        if isinstance(encode_type, str):
            encode_type = encode_type.lower()
        assert encode_type in [None, 'png', 'jpg', 'jpeg'], \
            ("`encode_type` should be in the list of"
             " [None, 'png', 'jpg', 'jpeg']!")

        self.info_built = False
        self.encode_type = encode_type
        self.quality = quality

    def add(self, data, label):
        """Add example.

        `data`: H * W (* C) uint8 numpy array
        `label`: int64 or float32 numpy array
        """
        assert data.dtype == np.uint8 and data.ndim in [2, 3], \
            '`data`: H * W (* C) uint8 numpy array!'

        if data.ndim == 2:
            data.shape = data.shape + (1,)

        if not self.info_built:
            self.data_shape = data.shape
            if self.encode_type is None:
                self.data_dtype_or_format = data.dtype.name
            else:
                self.data_dtype_or_format = self.encode_type
            self.label_shape = label.shape

        assert data.shape == self.data_shape, \
            'shapes of `data`s are inconsistent!'

        # tobytes
        if self.encode_type is not None:
            if data.ndim == 3:
                if data.shape[-1] == 1:
                    data.shape = data.shape[:2]
                elif data.shape[-1] != 3:
                    raise Exception('Only images with 1 or 3 '
                                    'channels are allowed to be encoded!')

            byte = io.BytesIO()
            data = Image.fromarray(data)
            if self.encode_type in ['jpg', 'jpeg']:
                data.save(byte, 'JPEG', quality=self.quality)
            elif self.encode_type == 'png':
                data.save(byte, 'PNG')
            data = byte.getvalue()

        super(ImageLablePairTfrecordCreator, self).add(data, label)


def tfrecord_batch(tfrecord_file, info_list, batch_size, preprocess_fns={},
                   shuffle=True, num_threads=16, min_after_dequeue=5000,
                   allow_smaller_final_batch=False, scope=None, compression_type=0):
    """Tfrecord batch ops.

    info_list:
        for example
        [{'name': 'img', 'decoder': tf.image.decode_png, 'decode_param': {}, 'shape': [112, 112, 1]},
         {'name': 'point', 'decoder': tf.decode_raw, 'decode_param': dict(out_type = tf.float32), 'shape':[136]}]

    preprocess_fns:
        for example
        {'img': img_preprocess_fn, 'point': point_preprocess_fn}
    """
    with tf.name_scope(scope, 'tfrecord_batch'):
        options = tf.python_io.TFRecordOptions(compression_type)

        data_num = 0
        for record in tf.python_io.tf_record_iterator(tfrecord_file, options):
            data_num += 1

        features = {}
        fields = []
        for info in info_list:
            features[info['name']] = tf.FixedLenFeature([], tf.string)
            fields += [info['name']]

        # read the next record (there is only one tfrecord file in the file queue)
        _, serialized_example = tf.TFRecordReader(options=options).read(
            tf.train.string_input_producer([tfrecord_file]))

        # parse the record
        features = tf.parse_single_example(serialized_example,
                                           features=features)

        # decode, set shape and preprocess
        data_dict = {}
        for info in info_list:
            name = info['name']
            decoder = info['decoder']
            decode_param = info['decode_param']
            shape = info['shape']

            feature = decoder(features[name], **decode_param)
            feature = tf.reshape(feature, shape)
            if name in preprocess_fns:
                feature = preprocess_fns[name](feature)
            data_dict[name] = feature

        # batch datas
        if shuffle:
            capacity = min_after_dequeue + (num_threads + 1) * batch_size
            data_batch = tf.train.shuffle_batch(
                data_dict,
                batch_size=batch_size,
                capacity=capacity,
                min_after_dequeue=min_after_dequeue,
                num_threads=num_threads,
                allow_smaller_final_batch=allow_smaller_final_batch)
        else:
            data_batch = tf.train.batch(
                data_dict,
                batch_size=batch_size,
                allow_smaller_final_batch=allow_smaller_final_batch)

        return data_batch, data_num, fields


class TfrecordData(object):
    """TfrecordData.

    preprocess_fns:
        for example
        {'img': img_preprocess_fn, 'point': point_preprocess_fn}
    """

    def __init__(self, tfrecord_path, batch_size, preprocess_fns={},
                 shuffle=True, num_threads=16, min_after_dequeue=5000,
                 allow_smaller_final_batch=False, scope=None):
        tfrecord_info_file = os.path.join(tfrecord_path, 'info.txt')
        tfrecord_file = os.path.join(tfrecord_path, 'data.tfrecord')

        with open(tfrecord_info_file) as f:
            try:  # for new version
                info_list = json.load(f)
            except:
                f.seek(0)
                info_list = ''
                for line in f.readlines():
                    info_list += line.strip('\n')
                info_list = eval(info_list)

        try:  # for new version
            for info in info_list:
                info['decoder'] = DECODERS[info['dtype_or_format']]['decoder']
                info['decode_param'] = \
                    DECODERS[info['dtype_or_format']]['decode_param']
        except:
            pass
        finally:
            if 'compression_type' in info_list[-1].keys():
                compression_type = info_list[-1]['compression_type']
                info_list[-1:] = []
            else:
                compression_type = 0

        self.graph = tf.Graph()  # declare ops in a separated graph
        with self.graph.as_default():
            # TODO
            # There are some strange errors if the gpu device is the
            # same with the main graph, but cpu device is ok. I don't know why...
            with tf.device('/cpu:0'):
                self._batch_ops, self._data_num, self._fields = \
                    tfrecord_batch(tfrecord_file, info_list, batch_size,
                                   preprocess_fns, shuffle, num_threads,
                                   min_after_dequeue, allow_smaller_final_batch,
                                   scope, compression_type)

        print(' [*] TfrecordData: create session!')

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess,
                                                    coord=self.coord)

    def __len__(self):
        return self._data_num

    def batch(self, fields=None):
        batch_data = self.sess.run(self._batch_ops)
        if fields is None:
            fields = self._fields
        if isinstance(fields, (list, tuple)):
            return [batch_data[field] for field in fields]
        else:
            return batch_data[fields]

    def fields(self):
        return self._fields

    def __del__(self):
        print(' [*] TfrecordData: stop threads and close session!')
        self.coord.request_stop()
        self.coord.join(self.threads)
        self.sess.close()

# Tfrecord Creator and Loader
- Convenient functions for converting datasets to tfrecord and loading it.

## Features
1. Support classification, regression and multiple label tasks.
1. Support image encoding (jpg or png) and decoding, which makes the tfrecord file smaller.
1. Support preprocessing functions for data and labels.

## Usage
- create a tfrecord and add a sample
    ```python
    writer = tfrecord.ImageLablePairTfrecordCreator(
        save_path='classification_tfrecord',
        encode_type='jpg',
        data_name='img',
        compression_type=0)

    writer.add(np.array(img), {"class": np.array(label)})
    ```

- load a tfrecord and get a batch
    ```python
    TR = tfrecord.TfrecordData(
        tfrecord_path='classification_tfrecord',
        batch_size=5,
        shuffle=True,
        num_threads=4)

    for i in range(iter):
        img_batch, class_batch = TR.batch(['img', 'class'])
    
    or

    for batch in TR:
        img_batch, class_batch = batch['img'], batch['class']
    ```
    - preprocessing
        ```python
        def img_preprecess_fn(img):
            img = tf.image.resize_images(img, [256, 128])
            img = img / 255
            return img

        TR = tfrecord.TfrecordData(
            tfrecord_path='classification_tfrecord',
            batch_size=5,
            shuffle=True,
            num_threads=4,
            preprocess_fns={
                'img': img_preprecess_fn
            })

        img_batch, class_batch = TR.batch(['img', 'class'])
        ```

 - More examples are in ***examples_create.py*** and ***examples_load.py***, just run it!

## Prerequisites
- tensorflow >= r1.2
- python 2.7

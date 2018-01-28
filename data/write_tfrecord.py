import os
import sys


import glob
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv

IMAGE_SIZE = 128

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  # return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(value)]))
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_image_binary(filename):
    """ You can read in the image using tensorflow too, but it's a drag
        since you have to create graphs. It's much easier using Pillow and NumPy
    """
    # image = Image.open(filename)
    # image = np.asarray(image, np.uint8)
    # shape = np.array(image.shape, np.int32)

    image = cv.imread(filename)
    image = cv.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    # cv.imshow('test',image)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # cv.waitKey()
    image = np.asarray(image, np.uint8)
    shape = np.array(image.shape, np.int32)
    print(shape)
    return shape.tobytes(), image.tostring()#image.tobytes() # convert image to raw data bytes in the array.


def write_to_tfrecord(writer, shape, binary_image, tfrecord_file):
    """ This example is to write a sample to TFRecord file. If you want to write
    more samples, just use a loop.
    """
    # writer = tf.python_io.TFRecordWriter(tfrecord_file)
    # write label, shape, and image content to the TFRecord file
    example = tf.train.Example(features=tf.train.Features(feature={
                'shape': _bytes_feature(shape),
                'image': _bytes_feature(binary_image)
                }))
    writer.write(example.SerializeToString())
    # writer.close()

# def write_tfrecord(label, image_file, tfrecord_file):
#     shape, binary_image = get_image_binary(image_file)
#     write_to_tfrecord(label, shape, binary_image, tfrecord_file)

def write_tfrecord(writer, image_file, tfrecord_file):
    shape, binary_image = get_image_binary(image_file)
    write_to_tfrecord(writer, shape, binary_image, tfrecord_file)



def read_from_tfrecord(reader, filenames):
    tfrecord_file_queue = tf.train.string_input_producer(filenames, name='queue')
    _, tfrecord_serialized = reader.read(tfrecord_file_queue)

    # label and image are stored as bytes but could be stored as
    # int64 or float64 values in a serialized tf.Example protobuf.
    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                        features={
                            'shape': tf.FixedLenFeature([], tf.string),
                            'image': tf.FixedLenFeature([], tf.string),
                        }, name='features')
    # image was saved as uint8, so we have to decode as uint8.
    image = tf.decode_raw(tfrecord_features['image'], tf.uint8)
    shape = tf.decode_raw(tfrecord_features['shape'], tf.int32)
    # the image tensor is flattened out, so we have to reconstruct the shape
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE,3])
    shape = tf.reshape(shape, [3])
    return image, shape

def read_vis_tfrecord(reader, tfrecord_file):
    image, shape = read_from_tfrecord(reader, [tfrecord_file])
    # init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        # sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(4):
            image_seq,shape_seq = sess.run([image,shape])
            print(tf.shape(shape_seq))
            print(shape_seq)
            # image_eval_reshape = image_eval.reshape([height.eval(), width.eval(), depth.eval()])
            pilimg = Image.fromarray(image_seq)
            print(pilimg.size)
            pilimg.show()
        coord.request_stop()
        coord.join(threads)
        sess.close()
        #
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(coord=coord)
        # image, shape = sess.run([image, shape])
        # coord.request_stop()
        # coord.join(threads)
    # print(shape)
    # print(image.shape)
    # plt.imshow(image)
    # plt.show()
    return image, shape

def read_tfrecord(reader, tfrecord_file):
    image, shape = read_from_tfrecord(reader, [tfrecord_file])

    # with tf.Session() as sess:
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #     image, shape = sess.run([image, shape])
    #     coord.request_stop()
    #     coord.join(threads)
    # print(shape)
    # plt.imshow(image)
    # plt.show()
    return image, shape

def main():
    paths = glob.glob('./images/*')
    tfrecord_filename = './test.tfrecord'
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)
    print('tf record writing start')
    # for path in paths:
    #     path = path + '/*'
    #     path = glob.glob(path)
    #     for subpath in path:
    #         print('image name: {}'.format(subpath))
    #         write_tfrecord(writer, subpath, tfrecord_filename)
    #         # print('done writing: {}'.format(subpath))
    # for path in paths:
    #     print(path)
    #     write_tfrecord(writer, path, tfrecord_filename)
    #     print('done writing: {}'.format(path))

    image_size = 128;
    for path in paths:
        path = path + '/*'
        path = glob.glob(path)
        for subpath in path:
            x=[]
            if 'curry' in subpath:
                print('image name: {}'.format(subpath))
                # write_tfrecord(writer, subpath,tfrecord_filename)
                write_tfrecord(writer, subpath,tfrecord_filename)

    writer.close()

    # testing tf record read1ng
    print('reading tf record')
    reader = tf.TFRecordReader()
    tfrecord_filename = './test.tfrecord'
    # closeshape, image = read_tfrecord(reader,tfrecord_filename)
    image, shape = read_vis_tfrecord(reader,tfrecord_filename)

    # print('shape:{}:'.format(shape))


# def __init__():
#     print('init')
if __name__ == '__main__':
    main()

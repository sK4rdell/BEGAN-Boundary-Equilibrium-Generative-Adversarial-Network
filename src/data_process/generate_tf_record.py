
import imageio
import time
import argparse
import glob
import tensorflow as tf
import numpy as np
from scipy.misc import imshow


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_images(paths):
    imgs = []
    for path in paths:
        imgs.append(imageio.imread(path))
    return imgs


def to_tf_records(data, name, id, store_dir):

    writer = tf.python_io.TFRecordWriter(
        "{}/{}{}".format(store_dir, name, str(id), '.tfrecords'))

    for xs in data:
        h, w, d = xs.shape  # shape of image
        image_raw = xs.tostring()
        # shape of annotation
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(h),
            'width': _int64_feature(w),
            'depth': _int64_feature(d),
            # store data
            'image_raw': _bytes_feature(image_raw),
        }))
        writer.write(example.SerializeToString())
    writer.close()


def create_tfrecords_data(img_dir, store_dir, max_imgs):

    # create training set
    paths = glob.glob(img_dir + '/*')
    num_samples = len(paths)
    num_train_sets = np.ceil(num_samples / max_imgs)

    # loop over samples and save to training-sets
    print('number of train imgs: ', len(paths))

    for j in range(int(num_train_sets)):
        if num_samples > (j + 1) * max_imgs:
            idx = np.arange(j * max_imgs, (j + 1) * max_imgs,
                            dtype=np.int32).tolist()
        else:
            idx = np.arange(j * max_imgs, num_train_sets,
                            dtype=np.int32).tolist()

        print('store training-set: :', str(j))
        train_imgs = load_images([paths[i] for i in idx])
        to_tf_records(train_imgs, 'training', j, store_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_dir', help='path to training images', type=str,
                        default='/home/simon/repos/generative_models/BEGAN/data/CelebA/images')
    parser.add_argument('--store_dir', help='path to store tfrecord files', type=str,
                        default='/home/simon/repos/generative_models/BEGAN/tfrecord_data')
    parser.add_argument('--max_size', help='max images in tfrecords-file',
                        type=int, default=2000)

    args = parser.parse_args()

    create_tfrecords_data(args.train_dir, args.store_dir, args.max_size)

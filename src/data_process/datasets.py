import tensorflow as tf


def read_and_decode(record, z_size):
    """ reads and decodes a tfrecord-file. """

    features = {'image_raw': tf.FixedLenFeature([], tf.string),
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'depth': tf.FixedLenFeature([], tf.int64)
                }

    parsed = tf.parse_single_example(record, features)

    height = tf.cast(parsed['height'], tf.int32)
    width = tf.cast(parsed['width'], tf.int32)
    depth = tf.cast(parsed['depth'], tf.int32)

    # shape of image and annotation
    #img_shape = tf.stack([height, width, depth])
    img_shape = tf.stack([64, 64, 3])

    # read, decode and normalize image
    image = tf.decode_raw(parsed['image_raw'], tf.uint8)
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    image = tf.reshape(image, img_shape)

    z = tf.random_uniform([z_size], minval=-1,
                          maxval=1, dtype=tf.float32)

    return image, z


def input_function(filenames, batch_size, epochs, z_size=128):

    def feeder():
        dataset = tf.data.TFRecordDataset(filenames)

        def parser(record):
            features, z = read_and_decode(record, z_size)
            # create latent vector z

            return features, z

        dataset = dataset.map(parser)
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(epochs)
        iterator = dataset.make_one_shot_iterator()

        features, z = iterator.get_next()
        return features, z
    return feeder


def feature_input_function(filenames, batch_size, epoch=None):
    return input_function(filenames, batch_size, epoch)


# Heade

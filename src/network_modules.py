import tensorflow as tf
import tensorflow.contrib as tfc


class Encoder(object):

    def __init__(self, depth, hidden, scope, strides=(2, 2)):
        self.depth = depth
        self.stride = strides
        self.scope = scope
        self.hidden_size = hidden

    def __call__(self, x, training, reuse=False):
        """ Creates the encoder. """
        print('SCOPE: ', self.scope)
        with tf.variable_scope(self.scope, reuse=reuse):
            # Feed Forward CNN

            x0 = tf.layers.conv2d(x, filters=self.depth, strides=self.stride, kernel_size=(5, 5),
                                  activation=tf.nn.elu,
                                  kernel_initializer=tf.variance_scaling_initializer,
                                  padding='same',
                                  name='encoder_conv1')

            x1 = Modules.higway_module(
                x0, (3, 3), 2, tf.nn.elu, 1, name='encoder_highway')

            # strided conv
            x2 = tf.layers.conv2d(x1,
                                  filters=self.depth,
                                  strides=self.stride,
                                  kernel_size=(3, 3),
                                  activation=tf.nn.elu,
                                  kernel_initializer=tf.variance_scaling_initializer,
                                  padding='same',
                                  name='encoder_conv2')

            x3 = Modules.higway_module(
                x2, (3, 3), 2, tf.nn.elu, 2, name='encoder_highway')

            x4 = tf.layers.conv2d(x3,
                                  filters=2 * self.depth,
                                  strides=self.stride,
                                  kernel_size=(3, 3),
                                  activation=tf.nn.elu,
                                  kernel_initializer=tf.variance_scaling_initializer,
                                  padding='same',
                                  name='encoder_conv3')
            x5 = Modules.higway_module(
                x4, (3, 3), 2, tf.nn.elu, 3, name='encoder_highway')

            x6 = tf.layers.conv2d(x5,
                                  filters=3 * self.depth,
                                  strides=self.stride,
                                  kernel_size=(3, 3),
                                  activation=tf.nn.elu,
                                  kernel_initializer=tf.variance_scaling_initializer,
                                  padding='same',
                                  name='encoder_conv4')
            x7 = Modules.higway_module(
                x6, (3, 3), 2, tf.nn.elu, 4, name='encoder_highway')

            x8 = tfc.layers.flatten(x7)

            h = tf.layers.dense(
                x8, self.hidden_size, kernel_initializer=tf.variance_scaling_initializer, name='enocer_dense')

            return h


class Decoder(object):

    def __init__(self, depth, scope, out_shape=64, strides=(2, 2),  num_channels=3):
        self.depth = depth
        self.stride = strides
        self.out_shape = out_shape
        self.scope = scope
        self.num_channels = num_channels

    def __call__(self, h, training, reuse=False):
        with tf.variable_scope(self.scope, reuse=reuse):
            map_size = int(self.out_shape / 3)
            num_units = map_size ** 2 * self.depth
            h0 = tf.layers.dense(
                h, num_units, kernel_initializer=tf.variance_scaling_initializer)

            # reshape to 3D
            h0 = tf.reshape(h0, shape=[-1, map_size, map_size, self.depth])

            x0 = Modules.higway_module(
                h0, (3, 3), 2, tf.nn.elu, 0, name='decoder_highway')

            map_size *= 2
            new_shape = [map_size, map_size]
            x0 = tf.image.resize_images(x0, new_shape,
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                        align_corners=False)

            x1 = Modules.higway_module(
                x0, (3, 3), 2, tf.nn.elu, 1, name='decoder_highway')

            new_shape = [self.out_shape, self.out_shape]
            x1 = tf.image.resize_images(
                x1, new_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False)

            x2 = Modules.higway_module(
                x1, (3, 3), 2, tf.nn.elu, 2, name='decoder_highway')

            x3 = tf.layers.conv2d(x2, filters=self.num_channels, strides=1, kernel_size=[8, 8],
                                  activation=tf.nn.tanh,
                                  kernel_initializer=tf.variance_scaling_initializer,
                                  padding='same')

            img = tf.layers.conv2d(x3, filters=self.num_channels, strides=1, kernel_size=[3, 3],
                                   activation=tf.nn.tanh,
                                   kernel_initializer=tf.variance_scaling_initializer,
                                   padding='same')
            return .5 * img


class AutoEncoder(object):

    def __init__(self, depth, scope, out_shape=64, strides=(2, 2), num_channels=3):
        self.depth = depth
        self.strides = strides
        self.out_shape = out_shape
        self.scope = scope
        self.num_channels = num_channels
        self.encoder = Encoder(depth, strides, scope)
        self.decoder = Decoder(depth, scope, out_shape,
                               strides, num_channels)

    def __call__(self, x, is_training, reuse=False):

        with tf.name_scope('Encoder'):
            h = self.encoder(x, is_training, reuse)
        with tf.name_scope('Decoder'):
            img = self.decoder(h, is_training, reuse)

        return img


class Modules:

    def higway_module(x, kernel_size, num_highway_units, activation, module_id,
                      name="highway_module"):
        """ Creates a set of connected highway layers.

        Parameters:
        ----------
        x : Tensor
            input
        kernel_size : List<int, int>
            size of kernel
        num_highway_units : int
            number of highway layers

        activation : Tensorflow OP.
            actiovation function
        module_id : int
            identifier
        name : string, optional
            name for the module (the default is "highway_module", which [default_description])

        Returns
        -------
        Tensor
            output from highway module
        """

        name += str(module_id)
        for i in range(num_highway_units):
            x = Modules.highway_unit(x, kernel_size, i, activation, name)
        return x

    def highway_unit(x, kernel_size, i, activation=tf.nn.relu,
                     name="highway_unit"):

        name += str(i)
        with tf.name_scope(name):
            num_maps = x.get_shape().as_list()[-1]

            H = tf.layers.conv2d(x, num_maps, kernel_size, activation=activation,
                                 kernel_initializer=tf.variance_scaling_initializer,
                                 padding='SAME',
                                 name=name + '_H')

            T = tf.layers.conv2d(x, num_maps, kernel_size,
                                 activation=tf.nn.sigmoid,
                                 kernel_initializer=tf.variance_scaling_initializer,
                                 padding='SAME',
                                 name=name + '_T')

            output = H * T + x * (1. - T)

        return output


class Losses:
    @staticmethod
    def BEGAN_loss(x, x_rec, gen_x, gen_x_rec, k, gamma=.5, lmbda=1e-3):

        rec_loss_x = tf.reduce_mean(tf.abs(x - x_rec))
        generator_loss = tf.reduce_mean(tf.abs(gen_x - gen_x_rec))
        disc_loss = rec_loss_x - k * generator_loss
        k_new = k + lmbda * (gamma * rec_loss_x - generator_loss)
        k_new = tf.clip_by_value(k_new, 0., 1.)
        k_assign_op = k.assign(k_new)
        # convergence meassure
        m = rec_loss_x + tf.abs(gamma * rec_loss_x - generator_loss)
        return generator_loss, disc_loss, m, k_assign_op

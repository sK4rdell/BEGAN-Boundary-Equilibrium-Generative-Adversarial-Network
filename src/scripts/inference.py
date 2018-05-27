__authur__ = "Simon Kardell"
'''
Inspired by the following blogpost: 
https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
'''
import argparse
import tensorflow as tf
import numpy as np
from google.protobuf import text_format
from scipy.misc import imshow, imsave, imread, imresize
from scipy.interpolate import griddata


def load_graph(graph_filename):

    graph_def = tf.GraphDef()

    with tf.gfile.GFile(graph_filename, 'rb') as g:
        data = g.read()
        graph_def.ParseFromString(data)

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='prefix')

    return graph


def create_patchwork(imgs, num_w, num_h):

    patch_work = np.concatenate(imgs[0:num_w], axis=1)
    k = num_w
    for i in range(num_h - 1):
        print(k)
        sub_patch = np.concatenate(imgs[k:k + num_w], axis=1)
        k += num_w
        patch_work = np.concatenate((patch_work, sub_patch), axis=0)

    return patch_work


def iterpolate_latent_vector(a, b, bins):

    sequence = a
    delta = (b - a) / bins
    for i in range(bins - 1):
        a += delta
        sequence = np.concatenate((sequence, a), axis=0)
    return sequence


def main(args):

    graph = load_graph(args.pb_path)

    # We access the input and output nodes
    x = graph.get_tensor_by_name('prefix/IteratorGetNext:1')
    y = graph.get_tensor_by_name('prefix/Generator/mul:0')
    x_disc = graph.get_tensor_by_name('prefix/IteratorGetNext:0')
    y_disc = graph.get_tensor_by_name(
        'prefix/Encoder/Discriminator/enocer_dense/BiasAdd:0')

    # We launch a Session
    with tf.Session(graph=graph) as sess:

        # Note: we don't nee to initialize/restore anything
        # There is no Variables in this graph, only hardcoded constants
        z = np.random.standard_normal([225, 64]) * .5
        y_out = sess.run(y, feed_dict={
            x: z  # < 45
        })
        img = create_patchwork(y_out, 15, 15)
        # I taught a neural net to recognise when a sum of numbers is bigger than 45
        # it should return False in this case
        imshow(img)
        imsave('patchwork.png', img)
        z1 = np.random.standard_normal([1, 64])
        z2 = np.random.standard_normal([1, 64])
        z = iterpolate_latent_vector(z1, z2, 10) * .5
        y_out = sess.run(y, feed_dict={x: z})

        interpolated_img = create_patchwork(y_out, 10, 1)
        imshow(interpolated_img)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pb_path", default="./training/frozen_model.pb",
                        type=str, help="Frozen model file to import")
    args = parser.parse_args()
    main(args)

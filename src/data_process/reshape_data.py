import argparse
import imageio
import numpy
import glob
import re
import os
import numpy as np
from scipy.misc import imresize
from filter import get_sorted_paths


def resize_images(dir_path, img_h, img_w):

    img_paths = get_sorted_paths(dir_path)
    print(len(img_paths))
    i = 0
    for path in img_paths:
        i += 1
        img = imageio.imread(path)
        img = imresize(img, [img_h, img_w, 3], interp='bilinear')
        imageio.imsave(path, img)
        if i % 1000 == 0:
            print('reshaped and saved {} images'.format(i))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', help='path to training images', type=str,
                        default='/home/simon/repos/generative_models/BEGAN/data/CelebA/images')
    parser.add_argument('--img_h', help='new image height',
                        type=int, default=64)
    parser.add_argument('--img_w', help='new image height',
                        type=int, default=64)

    args = parser.parse_args()

    resize_images(args.data_dir, args.img_h, args.img_w)

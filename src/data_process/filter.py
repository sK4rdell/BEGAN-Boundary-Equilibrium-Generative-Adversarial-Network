import argparse
import imageio
import numpy
import glob
import re
import os
import numpy as np
from scipy.misc import imresize


def get_int_in_string(name):
    number = int(re.search(r'\d+', name).group())
    return number


def get_sorted_paths(dir_path):
    paths = glob.glob("{}/*".format(dir_path))
    paths.sort(key=get_int_in_string)
    return paths


def filter(img_dir, label_dir):
    img_paths = get_sorted_paths(img_dir)
    annotation_paths = get_sorted_paths(label_dir)
    for img_path, anno_path in zip(img_paths, annotation_paths):
        annotation = imageio.imread(anno_path)
        img = imageio.imread(img_path)
        # all instances in image
        instances = np.unique(annotation[:, :, 1]).tolist()
        instances.remove(0)
        # If there's no annotaitons avaliable or no color-channel remove sample from dataset
        if not np.sum(instances) or img.ndim < 3:
            print('removing{}\n{}'.format(img_path, anno_path))
            os.remove(anno_path)
            os.remove(img_path)


def main(train_dir, train_label_dir, val_dir, val_label_dir):
    filter(train_dir, train_label_dir)
    filter(val_dir, val_label_dir)


def remove_leafs(paths1, paths2):

    ints1 = []
    ints2 = []
    for p in paths1:
        ints1.append(get_int_in_string(p))
    for p in paths2:
        ints2.append(get_int_in_string(p))

    for i in ints1:
        if not i in ints2:
            print('index ', i, 'not in training set')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_dir', help='path to training images', type=str,
                        default='/home/simon/data/places_challenge/images/training')
    parser.add_argument('--val_dir', help='path to validation images', type=str,
                        default='/home/simon/data/places_challenge/images/validation')
    parser.add_argument('--train_labels', help='path to training annotations', type=str,
                        default='/home/simon/data/places_challenge/annotations_instance/training')
    parser.add_argument('--val_labels', help='path to validation annotations', type=str,
                        default='/home/simon/data/places_challenge/annotations_instance/validation')
    args = parser.parse_args()

    main(args.train_dir, args.train_labels, args.val_dir, args.val_labels)

"""
Path to dataset should have following structure:
root:
    object1_name:
        rgb:    # RGB images (.png)
        mask:   # segmentation mask grasyscale images (.png)
        depth:  # depth information (.exr)
        gt:     # object to camera transformation (.json) - UNNECESSARY
    object2_name:
        ...
"""

import sys
import argparse
import glob
import os
import random
from pathlib import Path

if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2
import numpy as np
from tqdm import tqdm
import h5py

import util


def get_example_paths(dataset_root, split, shuffle=True):
    object_examples = {}

    for obj_base_dir in glob.glob(args.dataset_root + '/*/'):
        obj_name = Path(obj_base_dir).stem
        example_numbers = []

        for obj_data_dir in glob.glob(obj_base_dir + '/mask/*'):
            example_number = Path(obj_data_dir).stem
            example_numbers.append(example_number)

        if shuffle:
            random.shuffle(example_numbers)

        split_index = int(len(example_numbers)*split)
        train_examples = example_numbers[:split_index]
        test_examples = example_numbers[split_index:]

        object_examples.update({
            obj_name: {
                'train': train_examples,
                'test': test_examples
            }
        })

    return object_examples


def create_dataset(example_paths, root):
    for obj_name in example_paths:

        train_example_paths = tqdm(example_paths[obj_name]['train'])
        test_example_paths = tqdm(example_paths[obj_name]['test'])

        train_example_paths.set_description('Generating training data...')
        test_example_paths.set_description('Generating test data...')

        train_data, train_label, train_seg = [], [], []
        test_data, test_label, test_seg = [], [], []

        for example_path in train_example_paths:
            data, label, seg = create_single_data(example_path, obj_name, root)

            train_data.append(data)
            train_label.append(label)
            train_seg.append(seg)

        for example_path in test_example_paths:
            data, label, seg = create_single_data(example_path, obj_name, root)

            test_data.append(data)
            test_label.append(label)
            test_seg.append(seg)

    dataset_train = {
        'data': np.array(train_data), 
        'label': np.array(train_label), 
        'seg': np.array(train_seg)
    }

    dataset_test = {
        'data': np.array(test_data), 
        'label': np.array(test_label), 
        'seg': np.array(test_seg)
    }

    return dataset_train, dataset_test


def create_single_data(example_path, obj_name, root):
    depth, mask = load_inputs(example_path, obj_name, root)
    (depth, mask), offsets = preprocess(depth, mask)

    X, Y, Z, mask = generate_pointcloud(
        depth, mask, 
        fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
        offsets = offsets,
        points_limit=args.points_limit,
        scaling_factor=args.scaling_factor)
    
    data = np.vstack((X, Y, Z)).T
    label = np.array([1])  # TODO remove hardcoded
    seg = mask

    return data, label, seg


def load_inputs(example_path, obj_name, root):
    depth_filename = os.path.join(root, obj_name, 'depth', '{}.exr'.format(example_path))
    mask_filename = os.path.join(root, obj_name, 'mask', '{}.png'.format(example_path))

    depth = cv2.imread(depth_filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]
    mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)

    return depth, mask


def preprocess(depth, mask):
    bbox = util.get_bbox_from_mask(mask)
    return util.crop_data(depth, mask, bbox), (bbox[0], bbox[1])


def generate_pointcloud(depth, mask, fx, fy, cx, cy, offsets, scaling_factor=1.0, points_limit=2048):
    X = np.tile(np.arange(depth.shape[1], dtype=np.float32) + offsets[0], (depth.shape[0], 1))
    Y = np.tile(np.arange(depth.shape[0], dtype=np.float32) + offsets[0], (depth.shape[1], 1)).T

    Z = depth / args.scaling_factor
    X = np.multiply(X-cx, Z) / fx
    Y = np.multiply(Y-cy, Z) / fy

    return limit_points(X, Y, Z, mask, points_limit=points_limit)


def limit_points(X, Y, Z, mask, points_limit=2048):
    random_pixels = np.random.choice(X.size, size=points_limit, replace=False)
    random_pixels = np.unravel_index(random_pixels, X.shape)

    return X[random_pixels], Y[random_pixels], Z[random_pixels], mask[random_pixels]


def write(dataset, filename):
    with h5py.File(filename, "w") as f:
        data = f.create_dataset('data', data=dataset['data'])
        label = f.create_dataset("label", data=dataset['label'])
        pid = f.create_dataset("pid", data=dataset['seg'])


def execute(args):

    example_paths = get_example_paths(args.dataset_root, split=args.train_test_split, shuffle=True)
    train_dataset, test_dataset = create_dataset(example_paths, root=args.dataset_root)

    write(train_dataset, 'ply_data_train0.h5')
    write(test_dataset, 'ply_data_test0.h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-root', '-d', help='path to dataset root')
    parser.add_argument('--output-format', '-o', choices=['ply', 'h5'], help='output file format')
    parser.add_argument('--fx', type=float, required=True, help='focal length x')
    parser.add_argument('--fy', type=float, required=True, help='focal length y')
    parser.add_argument('--cx', type=float, required=True, help='principal point x')
    parser.add_argument('--cy', type=float, required=True, help='principal point y')
    parser.add_argument('--scaling-factor', type=float, default=1.0, help='scaling factor')
    parser.add_argument('--points-limit', '-l', type=int, default=2048, help='points number limit')
    parser.add_argument('--train-test-split','-s', type=float, default=0.8, help='train test split')

    global args
    args = parser.parse_args()

    execute(args)
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

        train_data, train_label = [], []
        test_data, test_label = [], []

        for example_path in train_example_paths:
            data, label = create_single_data(example_path, obj_name, root)

            train_data.append(data)
            train_label.append(label)

        for example_path in test_example_paths:
            data, label = create_single_data(example_path, obj_name, root)

            test_data.append(data)
            test_label.append(label)

    dataset_train = {
        'data': np.array(train_data), 
        'label': np.array(train_label)
    }

    dataset_test = {
        'data': np.array(test_data), 
        'label': np.array(test_label)
    }

    return dataset_train, dataset_test


def create_single_data(example_path, obj_name, root):
    rgb, depth, mask = load_inputs(example_path, obj_name, root)
    (rgb, depth, mask), offsets = preprocess(rgb, depth, mask)

    (X, Y, Z, R, G, B, norm_X, norm_Y, norm_Z), mask = generate_pointcloud(
        rgb, depth, mask, 
        fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
        offsets = offsets,
        points_limit=args.points_limit,
        scaling_factor=args.scaling_factor)

    data = np.vstack((X, Y, Z, R, G, B, norm_X, norm_Y, norm_Z)).T
    label = mask

    return data, label


def load_inputs(example_path, obj_name, root):
    rgb_filename = os.path.join(root, obj_name, 'rgb', '{}.png'.format(example_path))
    depth_filename = os.path.join(root, obj_name, 'depth', '{}.exr'.format(example_path))
    mask_filename = os.path.join(root, obj_name, 'mask', '{}.png'.format(example_path))

    rgb = cv2.cvtColor(cv2.imread(rgb_filename), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]
    mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)

    return rgb, depth, mask


def preprocess(rgb, depth, mask):
    bbox = util.get_bbox_from_mask(mask)
    bbox = list(bbox)
    bbox[0] = np.maximum(0, bbox[0]-50)
    bbox[1] = np.maximum(0, bbox[1]-50)
    bbox[2] = np.minimum(rgb.shape[1]-1, bbox[2]+50)
    bbox[3] = np.minimum(rgb.shape[0]-1, bbox[3]+50)
    return util.crop_data(rgb=rgb, depth=depth, mask=mask, bbox=bbox), (bbox[0], bbox[1])


def generate_pointcloud(rgb, depth, mask, fx, fy, cx, cy, offsets, scaling_factor=1.0, points_limit=4096):
    X = np.tile(np.arange(depth.shape[1], dtype=np.float32) + offsets[0], (depth.shape[0], 1))
    Y = np.tile(np.arange(depth.shape[0], dtype=np.float32) + offsets[0], (depth.shape[1], 1)).T

    Z = depth / args.scaling_factor
    X = np.multiply(X-cx, Z) / fx
    Y = np.multiply(Y-cy, Z) / fy

    indices = choose_indices(X.shape, limit=points_limit)

    X = X[indices]
    Y = Y[indices]
    Z = Z[indices]
    R = rgb[:, :, 0][indices] / 255.0
    G = rgb[:, :, 1][indices] / 255.0 
    B = rgb[:, :, 2][indices] / 255.0
    norm_X = normalize(X)
    norm_Y = normalize(Y)
    norm_Z = normalize(Z)

    mask = mask[indices]

    # TODO check normalization method 

    return (X, Y, Z, R, G, B, norm_X, norm_Y, norm_Z), mask


def normalize(x):
    min, max = np.min(x), np.max(x)
    return (x-min) / (max-min)


def choose_indices(shape, limit=4096):
    random_pixels = np.random.choice((shape[0] * shape[1]), size=limit, replace=False)
    random_pixels = np.unravel_index(random_pixels, shape)

    return random_pixels


def write(dataset, filename):
    with h5py.File(filename, "w") as f:
        data = f.create_dataset('data', data=dataset['data'])
        label = f.create_dataset("label", data=dataset['label'])


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
    parser.add_argument('--points-limit', '-l', type=int, default=4096, help='points number limit')
    parser.add_argument('--train-test-split','-s', type=float, default=0.8, help='train test split')

    global args
    args = parser.parse_args()

    execute(args)
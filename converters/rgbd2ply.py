import argparse
import sys
import os
if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2
import imageio
import numpy as np
import writer


def read_inputs(rgb_file, depth_file, mask_file):
    """
    Read RGB and depth files.

    Parameters:
    rgb_file (str): RGB image filename (format: .png)
    depth_file (str): RGB image filename (format: .exr)
    
    Returns
    tuple (numpy.ndarray, numpy.ndarray): rgb and depth
    """
    rgb, depth = None, None

    if rgb_file is not None:
        rgb = cv2.imread(args.rgb)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(args.depth, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

    if args.rgb is not None and (rgb.shape[:2] != depth.shape):
        raise Exception("Color and depth image do not have the same resolution.")

    return rgb, depth, mask


def crop_inputs(rgb, depth, mask):
    """
    Crops RGB and depth image to leave only object data. 
    Cropping is done with bounding box from segmentation mask.

    Parameters:
    rgb (numpy.ndarray [h x w x 3]) RGB image
    depth (numpy.ndarray [h x w]) depth image
    mask (numpy.ndarray [h x w]) segmentation mask image

    Returns:
        (numpy.ndarray [cropped_h x cropped_w x 3]) cropped RGB
        (numpy.ndarray [cropped_h x cropped_w]) cropped depth
    """
    x1, y1, x2, y2 = bbox_from_mask(mask)

    if rgb is not None:
        rgb = rgb[y1:y2, x1:x2, :]
    depth = depth[y1:y2, x1:x2]
    mask = mask[y1:y2, x1:x2]

    return rgb, depth, mask


def bbox_from_mask(mask):
    """
    Extract bounding box from segmentation mask. Every 
    positive value pixel is regarded as mask pixel.

    Parameters:
    mask (numpy.ndarray) one-channel segmenation mask

    Returns:
        (int, int, int, int): bounding box in format (x1, y1, x2, y2)
    """
    mask_pixels = np.where(mask > 0)

    if len(mask_pixels[0]) == 0:
        return -1, -1, -1, -1

    bbox_x1 = int(np.min(mask_pixels[1]))
    bbox_y1 = int(np.min(mask_pixels[0]))
    bbox_x2 = int(np.max(mask_pixels[1]))
    bbox_y2 = int(np.max(mask_pixels[0]))

    return bbox_x1, bbox_y1, bbox_x2, bbox_y2


def generate_pointcloud(args):
    """
    Generate point cloud in PLY format from depth and optionally rgb image.
    
    Parameters
    args (argparse.Namespace): cmd line arguments for pointcloud generation 
    """
    rgb, depth, mask = read_inputs(args.rgb, args.depth, args.mask)
    rgb, depth, mask = crop_inputs(rgb, depth, mask)

    X = np.tile(np.arange(depth.shape[1]), (depth.shape[0], 1))
    Y = np.tile(np.arange(depth.shape[0]), (depth.shape[1], 1)).T

    Z = depth / args.scaling_factor
    X = np.multiply(X-args.cx, Z) / args.fx
    Y = np.multiply(Y-args.cy, Z) / args.fy

    writer.write(X, Y, Z, 
        rgb=rgb,
        mask=mask, 
        output_filename=args.output
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--rgb', '-r', help='input RGB image - optional (format: .png)')
    parser.add_argument('--depth', '-d', required=True, help='input depth image (format: .exr)')
    parser.add_argument('--mask', '-m', required=True, help='input mask image (format: .png)')
    parser.add_argument('--output', '-o', required=True, help='output pointcloud file (format: .ply)')
    parser.add_argument('--fx', type=float, required=True, help='focal length x')
    parser.add_argument('--fy', type=float, required=True, help='focal length y')
    parser.add_argument('--cx', type=float, required=True, help='principal point x')
    parser.add_argument('--cy', type=float, required=True, help='principal point y')
    parser.add_argument('--scaling-factor', type=float, default=1000.0, help='scaling factor')

    args = parser.parse_args()

    generate_pointcloud(args)
    

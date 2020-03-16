import argparse
import sys
import os
if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2
import imageio
import numpy as np


def write_ply(points, output_filename, use_rgb=False):
    """
    Writes points of pointcloud to .ply file

    Parameters:
    points (numpy.ndarray): pointcloud points
    output_filename (str): output filename
    use_rgb (bool): is pointcloud RGBD or depth only
    """
    file = open(output_filename, "w")
    
    file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
%s
property uchar alpha
end_header
%s
'''%(len(points), '''property uchar red
property uchar green
property uchar blue
''' if use_rgb else '', "".join(points)))
    
    file.close()


def read_inputs(rgb_file, depth_file):
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

    if args.rgb is not None and (rgb.shape[:2] != depth.shape):
        raise Exception("Color and depth image do not have the same resolution.")

    return rgb, depth


def generate_pointcloud(args):
    """
    Generate point cloud in PLY format from depth and optionally rgb image.
    
    Parameters
    args (argparse.Namespace): cmd line arguments for pointcloud generation 
    """
    import time
    start_time = time.time()
    rgb, depth = read_inputs(args.rgb, args.depth)

    points = []

    X = np.arange(depth.shape[0])
    Y = np.arange(depth.shape[1])
    X = np.tile(X, (depth.shape[1], 1))
    Y = np.tile(Y, (depth.shape[0], 1)).T

    Z = depth / args.scaling_factor
    X = np.multiply(X-args.cx, Z) / args.fx
    Y = np.multiply(Y-args.cy, Z) / args.fy

    for v in range(depth.shape[1]):
        for u in range(depth.shape[0]):
            if Z[u, v] == 0:
                continue
            
            if rgb is not None:
                points.append("%f %f %f %d %d %d 0\n"%(
                    X[u, v], Y[u, v], Z[u, v], 
                    rgb[u, v, 0], rgb[u, v, 1], rgb[u, v, 2]))
            else:
                points.append("%f %f %f 0\n"%(X[u, v], Y[u, v], Z[u, v]))
    print('Conversion took {} seconds'.format(time.time() - start_time))

    write_ply(points, args.output, use_rgb=args.rgb != None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--rgb', '-r', help='input RGB image - optional (format: .png)')
    parser.add_argument('--depth', '-d', required=True, help='input depth image (format: .exr)')
    parser.add_argument('--output', '-o', required=True, help='output pointcloud file (format: .ply)')
    parser.add_argument('--fx', type=float, required=True, help='focal length x')
    parser.add_argument('--fy', type=float, required=True, help='focal length y')
    parser.add_argument('--cx', type=float, required=True, help='principal point x')
    parser.add_argument('--cy', type=float, required=True, help='principal point y')
    parser.add_argument('--scaling-factor', type=float, default=1000.0, help='scaling factor')

    args = parser.parse_args()

    generate_pointcloud(args)
    

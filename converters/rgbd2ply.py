import argparse
import sys
import os
if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2
import imageio


def write_ply(points, output_filename, use_rgb=False):
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


def generate_pointcloud(args):
    """
    Generate point cloud in PLY format from depth and optionally rgb image.
    
    Parameters
    args (argparse.Namespace): cmd line arguments for pointcloud generation 
    """
    if args.rgb is not None:
        rgb = cv2.imread(args.rgb)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(args.depth, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]

    if args.rgb is not None and (rgb.shape[:2] != depth.shape):
        raise Exception("Color and depth image do not have the same resolution.")

    points = []    
    for v in range(depth.shape[1]):
        for u in range(depth.shape[0]):
            if depth[u, v] == 0:
                continue
            
            Z = depth[u, v] / args.scaling_factor
            X = (u - args.cx) * Z / args.fx
            Y = (v - args.cy) * Z / args.fy
            
            if args.rgb is not None:
                color = rgb[u, v]
                points.append("%f %f %f %d %d %d 0\n"%(X, Y, Z, color[0], color[1], color[2]))
            else:
                points.append("%f %f %f 0\n"%(X, Y, Z))

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
    

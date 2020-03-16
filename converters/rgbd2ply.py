import argparse
import sys
import os
if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2
import imageio


def generate_pointcloud(args):
    """
    Generate point cloud in PLY format from depth and optionally rgb image.
    
    Parameters
    args (argparse.Namespace): cmd line arguments for pointcloud generation 
    """
    rgb = cv2.imread(args.rgb)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(args.depth, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]

    if rgb.shape[:2] != depth.shape:
        raise Exception("Color and depth image do not have the same resolution.")

    points = []    
    for v in range(rgb.shape[1]):
        for u in range(rgb.shape[0]):
            color = rgb[u, v]
            Z = depth[u, v] / args.scaling_factor

            if Z == 0: continue
            
            X = (u - args.cx) * Z / args.fx
            Y = (v - args.cy) * Z / args.fy
            
            points.append("%f %f %f %d %d %d 0\n"%(X, Y, Z, color[0], color[1], color[2]))
    
    file = open(args.output, "w")
    file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
%s
'''%(len(points),"".join(points)))
    file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--rgb', '-r', help='input RGB image (format: .png)')
    parser.add_argument('--depth', '-d', help='input depth image (format: .exr)')
    parser.add_argument('--output', '-o', help='output pointcloud file (format: .ply)')
    parser.add_argument('--fx', type=float, help='focal length x')
    parser.add_argument('--fy', type=float, help='focal length y')
    parser.add_argument('--cx', type=float, help='principal point x')
    parser.add_argument('--cy', type=float, help='principal point y')
    parser.add_argument('--scaling-factor', type=float, help='scaling factor')
    # TODO put camera intrinsics to config file

    args = parser.parse_args()

    generate_pointcloud(args)
    

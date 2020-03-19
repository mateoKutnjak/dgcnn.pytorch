"""
Visualizes .ply file
"""


import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')

    args = parser.parse_args()

    if args.filename.endswith('.ply'):
        import open3d

        pcd = open3d.io.read_point_cloud(args.filename)
        open3d.visualization.draw_geometries([pcd])
    elif args.filename.endswith('.h5'):
        import h5py
        import torch
        import visualization
        import numpy as np
        import random
        import matplotlib.pyplot as pyplot
        from mpl_toolkits.mplot3d import Axes3D

        with h5py.File(args.filename, 'r') as f:
            it = list(range(len(f['data'])))
            random.shuffle(it)

            for i in it:    
                points = torch.from_numpy(np.transpose(f['data'][i:(i+1)], (0, 2, 1))).float().to('cuda')
                seg = torch.from_numpy(f['pid'][i:(i+1)]).float().to('cuda')

                visualization.plot_partseg(points, seg)

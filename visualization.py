import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


colors = ['r', 'g', 'b', 'y']


def plot_partseg(pointcloud, seg):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = pointcloud[0, :, 0]
    y = pointcloud[0, :, 1]
    z = pointcloud[0, :, 2]

    for i, c in enumerate(np.unique(seg[0])):
        indices = np.where(seg[0] == c)[0]
        ax.scatter(x[indices], y[indices], z[indices], c=colors[i], marker='o')

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0

    mid_x = (x.max()+ x.min()) * 0.5
    mid_y = (y.max()+ y.min()) * 0.5
    mid_z = (z.max()+ z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()

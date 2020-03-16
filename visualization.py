import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


colors = ['r', 'g', 'b', 'y']


def plot_partseg(pointcloud, seg):
    """
    Plots pointcloud and colors each point based on points segmentation class.

    Parameters:
    pointcloud (torch.Tensor [batch_size x 3 x num_points]): pointcloud
    seg (torch.Tensor [batch_size x num_points]): segmentation classes for each point
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    _pointcloud = pointcloud.cpu().detach().numpy()
    _seg = seg.cpu().detach().numpy()

    x = _pointcloud[0, 0, :]
    y = _pointcloud[0, 1, :]
    z = _pointcloud[0, 2, :]

    for i, c in enumerate(np.unique(_seg[0])):
        indices = np.where(_seg[0] == c)[0]
        ax.scatter(x[indices], y[indices], z[indices], c=colors[i], marker='o')

    preserve_ratio(ax, x=x, y=y, z=z)

    plt.show()


def preserve_ratio(ax, x, y, z):
    """
    Based on (x,y,z) coordinates of pointcloud, limits of matplotlib Axes are set
    to preserve scale of 3D plot

    Parameters:
    ax (matplotlib.axes._subplots.Axes3DSubplot): matplotlib Axes object
    x (numpy.ndarray [num_points,]): x coordinates of pointcloud
    y (numpy.ndarray [num_points,]): y coordinates of pointcloud
    z (numpy.ndarray [num_points,]): z coordinates of pointcloud
    """
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0

    mid_x = (x.max()+ x.min()) * 0.5
    mid_y = (y.max()+ y.min()) * 0.5
    mid_z = (z.max()+ z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
import numpy as np
import h5py


def write(X, Y, Z, **kwargs):
    if kwargs.get('output_filename').endswith('.ply'):
        write_ply(X, Y, Z, **kwargs)
    elif kwargs.get('output_filename').endswith('.h5'):
        write_h5(X, Y, Z, **kwargs)
    else:
        raise Exception("Unsupported pointcloud output type.")


def write_h5(points, mask, output_filename):
    pass
    # f = h5py.File(output_filename, "w")
    
    # data = f.create_group("data")
    # label = f.create_group("label")
    # seg = f.create_group("seg")

    # import pdb
    # pdb.set_trace()


def write_ply(X, Y, Z, **kwargs):
    points = []

    rgb = kwargs.get('rgb')

    if rgb is not None:
        for x, y, z, r, g, b in np.nditer([X, Y, Z, rgb[..., 0], rgb[..., 1], rgb[..., 2]]):
            points.append("%f %f %f %d %d %d 0\n"%(x, y, z, r, g, b))
    else:
        for x, y, z in np.nditer([X, Y, Z]):
            points.append("%f %f %f 0\n"%(x, y, z))

    """
    Writes points of pointcloud to .ply file

    Parameters:
    points (numpy.ndarray): pointcloud points
    output_filename (str): output filename
    use_rgb (bool): is pointcloud RGBD or depth only
    """
    file = open(kwargs.get('output_filename'), "w")
    
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
''' if rgb is not None else '', "".join(points)))
    
    file.close()
import numpy as np
import h5py


def write(X, Y, Z, point_num=2048, **kwargs):
    if X.size > point_num:
        random_pixels = np.random.choice(X.shape[0]*X.shape[1], size=point_num, replace=False)
        random_pixels = np.unravel_index(random_pixels, X.shape)

        X = X[random_pixels]
        Y = Y[random_pixels]
        Z = Z[random_pixels]
        if kwargs.get('rgb') is not None:
            kwargs['rgb'] = kwargs.get('rgb')[random_pixels]
        kwargs['mask'] = kwargs.get('mask')[random_pixels]

    if kwargs.get('output_filename').endswith('.ply'):
        write_ply(X, Y, Z, point_num, **kwargs)
    elif kwargs.get('output_filename').endswith('.h5'):
        write_h5(X, Y, Z, point_num,**kwargs)
    else:
        raise Exception("Unsupported pointcloud output type.")


def write_h5(X, Y, Z, point_num, **kwargs):
    with h5py.File(kwargs.get('output_filename'), "w") as f:
        _data = np.vstack((X, Y, Z)).T
        _data = np.expand_dims(_data, axis=0)
        _label = np.array([1])
        _label = np.expand_dims(_label, axis=0)
        _seg = kwargs.get('mask')
        _seg = np.expand_dims(_seg, axis=0)
        
        data = f.create_dataset('data', data=_data)
        label = f.create_dataset("label", data=_label)
        seg = f.create_dataset("seg", data=_seg)


def write_ply(X, Y, Z, point_num, **kwargs):
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
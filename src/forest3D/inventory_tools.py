import numpy as np
from forest3D import processLidar

def get_tree_tops(xyz_data,labels,percentile=99.9):
    '''

    :param xyz_data: np.array of size Nx3. N is number of points. Columns are x,y,z.
    :param labels: np.array of size N. Delineation labels output by detector
    :param percentile: float between 0 and 100, used to pick non-outlier highest point.
    :return: np.array of size Mx4, where M is number of trees. Tree-top (x,y,z coordinate) for each tree in labels.
            Columns are x_top,y_top,z_top,label_id
    '''

    label_id = np.unique(labels)
    tree_tops = np.zeros((len(label_id)-1,4))
    for i, label in enumerate(label_id):
        if label>0:
            xyz_tree = xyz_data[labels == label, :]
            tallest_idx = abs(xyz_tree[:,2] - np.percentile(xyz_tree[:,2], percentile, interpolation='nearest')).argmin()
            tree_tops[i - 1, :3] = xyz_tree[tallest_idx, :]
            tree_tops[i - 1, 3] = label

    return tree_tops


def get_tree_heights(tree_tops,ground_points):
    '''

    :param tree_tops: Mx3 np.array of x,y,z tree top coordinates
    :param ground_points: Px3 np.array of x,y,z ground point coordinates. Obtained using ground_removal.load_ground_surface()
    :return: np.array of size M of tree heights from ground
    '''

    PCD = processLidar.ProcessPC(tree_tops.copy())
    PCD.ground_normalise(ground_points)
    return PCD.pc[:, 2]
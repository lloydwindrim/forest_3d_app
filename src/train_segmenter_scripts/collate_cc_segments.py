import numpy as np
import glob

import sys
sys.path.insert(0, '../')
from forest3D import lidar_IO


for i in range(10):
    directory = 'path/to/trees/folder/tree%i_labels/*'%(i)
    filename_list = glob.glob( ( directory  ) )

    pcd = np.zeros((1,4))
    labels = np.zeros((1))

    for file in filename_list:
        xyz_list = lidar_IO.XYZreadFromCSV(file,delimiter=',',x=0,y=1,z=2,returns=3)
        if 'foliage' in file:
            labels = np.hstack( ( labels , 0*np.ones((xyz_list.shape[0])) ) )
            pcd = np.vstack(( pcd, xyz_list ))
        elif 'hgStem' in file:
            labels = np.hstack( ( labels , 1*np.ones((xyz_list.shape[0])) ) )
            pcd = np.vstack(( pcd, xyz_list ))
        elif 'lgStem' in file:
            labels = np.hstack( ( labels , 2*np.ones((xyz_list.shape[0])) ) )
            pcd = np.vstack(( pcd, xyz_list ))
        elif 'clutter' in file:
            labels = np.hstack((labels, 3 * np.ones((xyz_list.shape[0]))))
            pcd = np.vstack((pcd, xyz_list))

    pcd = pcd[1:,:]
    labels = labels[1:]

    lidar_IO.writeXYZ_labelled('path/to/trees/folder/tree%i_labels/tree%i_labelled.asc' % (i, i), pcd[:,:3],labels,returns=pcd[:,3])



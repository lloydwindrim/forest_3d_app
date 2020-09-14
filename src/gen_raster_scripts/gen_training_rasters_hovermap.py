import os
import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.decomposition import PCA
import pickle

import sys
sys.path.insert(0, '../')
from forest3D import lidar_IO,treeDetector,ground_removal,utilities,detection_tools


# size limitation: 120m x 120m

output_dir = '/home/lloyd/Documents/projects/forest_3d_app/outputs'

# read data
path_pcd = '/home/lloyd/Documents/datasets/lidar/hovermap_nzl/Interpine_02_Output_laz1_2_down5cm.las'
xyz_data = lidar_IO.readFromLas(path_pcd,fields = ['x','y','z','intensity'])
MAX_RETURN_INTENSITY = 77.0
xyz_data[:, 3] /= MAX_RETURN_INTENSITY

# remove ground, output pcd to help with labelling
offset = [0,0,0]
xyz_gr,colour_gr = ground_removal.removeGround(xyz_data[:,:3], returns=xyz_data[:,3], offset=offset, thresh=2.0,
                                               proc_path=output_dir, name='')
ground_pts = ground_removal.load_ground_surface(os.path.join(output_dir, '_ground_surface.ply'))
xyz_colour_gr = np.hstack((xyz_gr,colour_gr))

# configure raster
config_dict = {'raster_layers': ['vertical_density','mean_colour1'], 'support_window': [1, 1],
               'normalisation': 'rescale+histeq','doHisteq':[True,True],'res': 0.05, 'gridSize': [600, 600, 1000]}
rasterMaker = treeDetector.RasterDetector(**config_dict)

# read in training crop info from csv
path = '/home/lloyd/Documents/datasets/lidar/hovermap_nzl/training_locations.csv'
plot_details = utilities.read_csv(path,header_rows=[0])
num_plots = np.shape(plot_details['data'])[0]

for i in range(num_plots):

    x_centre = float(plot_details['data'][i,1])
    y_centre = float(plot_details['data'][i,2])
    radius = float(plot_details['data'][i,3])

    # extract pointcloud crop
    xyz_crop_gr = detection_tools.circle_crop(xyz_colour_gr,x_centre,y_centre,radius)


    # create raster
    raster = rasterMaker.rasterise(xyz_crop_gr[:,:3],colour_data=xyz_crop_gr[:,3],ground_pts=ground_pts)


    # save rasters
    filename = os.path.split(path)[1].split('.')[0]
    plt.imsave(os.path.join(output_dir,'raster_%i.jpg'%(i)), raster)

with open(os.path.join(output_dir,'raster_config.json'), 'w') as outfile:
    json.dump(config_dict, outfile)




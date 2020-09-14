import os
import matplotlib.pyplot as plt
import json
import numpy as np

import sys
sys.path.insert(0, '../')
from forest3D import lidar_IO,treeDetector,ground_removal,utilities,detection_tools


# size limitation: 120m x 120m

output_dir = '/home/lloyd/Documents/projects/forest_3d_app/outputs'

# read data
path_pcd = '/home/lloyd/Documents/datasets/lidar/low_res_ALS/saltwater_31B_1_2.las'
xyz_data = lidar_IO.readFromLas(path_pcd, fields = ['x','y','z'])

# remove ground, output pcd to help with labelling
offset = [0,0,0]
xyz_data_gr = ground_removal.removeGround(xyz_data, offset, thresh=2.0,proc_path=output_dir, name='')
ground_pts = ground_removal.load_ground_surface(os.path.join(output_dir, '_ground_surface.ply'))


# configure raster
config_dict = {'raster_layers': ['vertical_density', 'canopy_height'], 'support_window': [11, 7, 1],
               'normalisation': 'rescale+histeq','doHisteq':[False,True],'res': 0.1, 'gridSize': [600, 600, 1000]}
rasterMaker = treeDetector.RasterDetector(**config_dict)

for i in range(1,20):

    # read in training crop info from csv
    path = '/home/lloyd/Documents/datasets/lidar/low_res_ALS/training_plots/training_stems_%03d.csv'%i
    plot_details = utilities.read_csv(path,header_rows=[0])
    x_centre = float(plot_details['header'][1])
    y_centre = float(plot_details['header'][2])
    radius = float(plot_details['header'][3])
    gt_coords = plot_details['data'][:,1:3]

    # extract pointcloud crop
    xyz_crop_gr = detection_tools.circle_crop(xyz_data_gr,x_centre,y_centre,radius)


    # create raster
    raster,centre = rasterMaker.rasterise(xyz_crop_gr,ground_pts=ground_pts,returnCentre=True)


    # annotate raster with field measurement
    raster_coords = treeDetector.pcd2rasterCoords(gt_coords, config_dict['gridSize'], config_dict['res'],centre)
    max_intensity = np.max(raster)
    radius = 4
    for k in range(gt_coords.shape[0]):
        raster[raster_coords['row'][k] - radius:raster_coords['row'][k] + radius, raster_coords['col'][k] - radius:raster_coords['col'][k] + radius,:] = max_intensity


    # save rasters
    filename = os.path.split(path)[1].split('.')[0]
    plt.imsave(os.path.join(output_dir,'raster_'+filename+'.jpg'), raster)

with open(os.path.join(output_dir,'raster_config.json'), 'w') as outfile:
    json.dump(config_dict, outfile)




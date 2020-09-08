import os
import matplotlib.pyplot as plt
import json

import sys
sys.path.insert(0, '../')
from forest3D import lidar_IO,treeDetector,ground_removal


# size limitation: 120m x 120m

output_dir = '/home/lloyd/Documents/projects/forest_3d_app/outputs'


# configure raster
config_dict = {'raster_layers': ['vertical_density', 'canopy_height', 'mean_colour1'], 'support_window': [1, 1, 1],
               'normalisation': 'rescale+histeq','doHisteq':[True,True,True],'res': 0.2, 'gridSize': [600, 600, 1000]}
rasterMaker = treeDetector.RasterDetector(**config_dict)


for i in [1,3,4,5,6,8,10,12,13,15,16,17,20,21,22,25,26]:

    path = '/home/lloyd/Documents/datasets/lidar/forestry_usyd/extracts/plots_for_training/V1_Scanner1_161011_220153_crop%03d.asc'%(i)
    xyz_data = lidar_IO.XYZreadFromCSV(path,delimiter=' ',x=0,y=1,z=2,returns=8)
    MAX_RETURN_INTENSITY = 50000
    xyz_data[:,3] /= MAX_RETURN_INTENSITY


    offset = [0,0,0]

    # remove ground, output pcd to help with labelling
    xyz_data_gr,returns_gr = ground_removal.removeGround(xyz_data[:,:3],offset,returns=xyz_data[:,3],thresh=2.0,
                                                         proc_path=output_dir, name='')
    ground_pts = ground_removal.load_ground_surface(os.path.join(output_dir,'_ground_surface.ply'))


    # create raster
    raster = rasterMaker.rasterise(xyz_data_gr,colour_data=returns_gr,ground_pts=ground_pts)


    # save rasters
    filename = os.path.split(path)[1].split('.')[0]
    plt.imsave(os.path.join(output_dir,'raster_'+filename+'.jpg'), raster)

with open(os.path.join(output_dir,'raster_config.json'), 'w') as outfile:
    json.dump(config_dict, outfile)




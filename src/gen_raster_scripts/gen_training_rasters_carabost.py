import os
import matplotlib.pyplot as plt
import json

import sys
sys.path.insert(0, '../')
from forest3D import lidar_IO,treeDetector,ground_removal


# size limitation: 120m x 120m

output_dir = '/home/lloyd/Documents/projects/forest_3d_app/outputs'

plot_list = ['Site2_Plot_0_p_60m_fixed', 'Site2_Plot_1_p_60m_fixed', 'Site4_Plot_04_p_60m_fixed', 'Site4_Plot_05_p_60m_fixed',
             'Site4_Plot_6_p_60m_fixed', 'Site8_Plot_15_p_60m_fixed', 'Site8_Plot_16_p_60m_fixed', 'Site9_Plot_17_p_60m_fixed']


# configure raster
config_dict = {'raster_layers': ['vertical_density', 'canopy_height'], 'support_window': [1, 1, 1],
               'normalisation': 'rescale+histeq','doHisteq':[True,True],'res': 0.2, 'gridSize': [600, 600, 1000]}
rasterMaker = treeDetector.RasterDetector(**config_dict)


for i in range(8):

    path = '/home/lloyd/Documents/datasets/lidar/forestry_usyd/newSite/extracts/plots_for_training/%s.asc' % (plot_list[i])
    xyz_data = lidar_IO.XYZreadFromCSV(path,delimiter=' ',x=0,y=1,z=2)


    offset = [0,0,0]

    # remove ground, output pcd to help with labelling
    xyz_data_gr = ground_removal.removeGround(xyz_data,offset,thresh=2.0,
                                                         proc_path=output_dir, name='')
    ground_pts = ground_removal.load_ground_surface(os.path.join(output_dir,'_ground_surface.ply'))


    # create raster
    raster = rasterMaker.rasterise(xyz_data_gr,ground_pts=ground_pts)


    # save rasters
    filename = os.path.split(path)[1].split('.')[0]
    plt.imsave(os.path.join(output_dir,'raster_'+filename+'.jpg'), raster)

with open(os.path.join(output_dir,'raster_config.json'), 'w') as outfile:
    json.dump(config_dict, outfile)




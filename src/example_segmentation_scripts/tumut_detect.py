import sys
sys.path.insert(0, '../')
from forest3D import lidar_IO,ground_removal,treeDetector,inventory_tools,utilities

import os
import json
import numpy as np

output_dir = '/home/lloyd/Documents/projects/forest_3d_app/outputs'

print("reading the data into python...")
path = '/home/lloyd/Documents/datasets/lidar/forestry_usyd/extracts/plots_for_training/V1_Scanner1_161011_220153_crop026.asc'
xyz_data = lidar_IO.XYZreadFromCSV(path,delimiter=' ',x=0,y=1,z=2, returns=8)
#path = '/home/lloyd/Documents/datasets/lidar/forestry_usyd/V1_Scanner1_161011_220153_crop001.las'
#xyz_data = lidar_IO.readFromLas(path, fields = ['x','y','z','Intensity'])
MAX_RETURN_INTENSITY = 50000
xyz_data[:, 3] /= MAX_RETURN_INTENSITY


print("removing the ground...")
offset = [0,0,0]       # optional: use to make calculations stable on large values
xyz_data_gr,intensity_gr = ground_removal.removeGround(xyz_data[:,:3],offset,returns=xyz_data[:,3],thresh=2.0,proc_path=output_dir)
ground_pts = ground_removal.load_ground_surface(os.path.join(output_dir,'_ground_surface.ply'))



print("detecting trees...")
detector_addr = '/home/lloyd/Documents/projects/forest_3d_app/models/detection/tumut1'
with open(os.path.join(detector_addr, 'raster_config.json')) as json_file:
    config_dict = json.load(json_file)
rasterTreeDetector = treeDetector.RasterDetector(**config_dict )
labels = rasterTreeDetector.sliding_window(detector_addr,xyz_data_gr,ground_pts=ground_pts,windowSize = [100,100],stepSize = 80)


print("output trees as individual pointclouds...")
for i in list(np.unique(labels)):
    if i > 0:
        lidar_IO.writeXYZ_labelled(os.path.join(output_dir,'tree_%i.asc'%(i)), xyz_data_gr[labels==i,:],
                                   intensity_gr[labels==i], delimiter=',')
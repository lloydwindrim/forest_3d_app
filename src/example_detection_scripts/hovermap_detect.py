import sys
sys.path.insert(0, '../')
from forest3D import lidar_IO,ground_removal,treeDetector,inventory_tools,utilities

import os
import json
import numpy as np

output_dir = '/home/lloyd/Documents/projects/forest_3d_app/outputs'

print("reading the data into python...")
path = '/home/lloyd/Documents/datasets/lidar/hovermap_nzl/Interpine_02_Output_laz1_2_down5cm.las'
#path = '/home/lloyd/Documents/datasets/lidar/hovermap_dogPark/DogParkBackpack_class_elev_5cm.las'
xyz_data = lidar_IO.readFromLas(path, fields = ['x','y','z','Intensity'])
MAX_RETURN_INTENSITY = 77.0
xyz_data[:, 3] /= MAX_RETURN_INTENSITY


print("removing the ground...")
offset = [0,0,0]       # optional: use to make calculations stable on large values
xyz_data_gr,intensity_gr = ground_removal.removeGround(xyz_data[:,:3],offset,returns=xyz_data[:,3],thresh=2.0,proc_path=output_dir)
ground_pts = ground_removal.load_ground_surface(os.path.join(output_dir,'_ground_surface.ply'))



print("detecting trees...")
detector_addr = '/home/lloyd/Documents/projects/forest_3d_app/models/detection/hovermap2'
with open(os.path.join(detector_addr, 'raster_config.json')) as json_file:
    config_dict = json.load(json_file)
rasterTreeDetector = treeDetector.RasterDetector(**config_dict )
labels = rasterTreeDetector.sliding_window(detector_addr,xyz_data_gr,colour_data=intensity_gr,ground_pts=ground_pts,
                                           windowSize = [22,22],stepSize = 15,overlap_thresh=3)



print("outputing detection results...")
lidar_IO.writePly_labelled(os.path.join(output_dir,'detection.ply'),xyz_data_gr,labels,offset)


print("outputting inventory...")
tree_tops = inventory_tools.get_tree_tops(xyz_data_gr,labels)
utilities.write_csv(os.path.join(output_dir,'inventory.csv'),tree_tops,header='x,y,z,id')


import sys
sys.path.insert(0, '../')
from forest3D import lidar_IO,ground_removal,treeDetector,inventory_tools,utilities

import os

import json
import pickle
import numpy as np

output_dir = '/home/lloyd/Documents/projects/forest_3d_app/outputs'

print("reading the data into python...")
path = '/home/lloyd/Documents/datasets/lidar/utas/transect_large_vis_dense_down5cm.las'
xyz_data = lidar_IO.readFromLas(path, fields = ['x','y','z','red','green','blue'],convert_colours=True)
xyz_data[:,3:] /= 255.0


print("removing the ground...")
offset = [0,0,0]       # optional: use to make calculations stable on large values
xyz_gr,colour_gr = ground_removal.removeGround(xyz_data[:,:3], returns=xyz_data[:,3:], offset=offset, thresh=0.5,
                                               proc_path=output_dir, name='')
ground_pts = ground_removal.load_ground_surface(os.path.join(output_dir, '_ground_surface.ply'))


# apply pca
pca = pickle.load(open("/home/lloyd/Documents/projects/forest_3d_app/models/detection/utas/pca.pkl","rb"))
pc_gr = pca.transform(colour_gr)


print("detecting trees...")
detector_addr = '/home/lloyd/Documents/projects/forest_3d_app/models/detection/utas'
with open(os.path.join(detector_addr, 'raster_config.json')) as json_file:
    config_dict = json.load(json_file)
rasterTreeDetector = treeDetector.RasterDetector(**config_dict )
labels = rasterTreeDetector.sliding_window(detector_addr,xyz_gr,colour_data=pc_gr,ground_pts=ground_pts,windowSize = [25,25],
                                           stepSize = 23,overlap_thresh=1)



print("outputing detection results...")
lidar_IO.writePly_labelled(os.path.join(output_dir,'detection.ply'),xyz_gr,labels,offset)


print("outputting inventory...")
tree_tops = inventory_tools.get_tree_tops(xyz_gr,labels)
heights = inventory_tools.get_tree_heights(tree_tops[:,:3],ground_pts)
inventory = np.hstack((tree_tops,heights[:,np.newaxis]))
utilities.write_csv(os.path.join(output_dir,'inventory.csv'),inventory,header='x,y,z,id,height')
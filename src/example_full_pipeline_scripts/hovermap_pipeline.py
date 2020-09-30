import sys
sys.path.insert(0, '../')
from forest3D import lidar_IO,ground_removal,treeDetector,inventory_tools,utilities,detection_tools,stemSegmenter

import os
import json
import numpy as np

output_dir = '/home/lloyd/Documents/projects/forest_3d_app/outputs'

print("reading the data into python...")
path = '/home/lloyd/Documents/datasets/lidar/hovermap_dogPark/DogParkBackpack_class_elev_5cm.las'
xyz_data = lidar_IO.readFromLas(path, fields = ['x','y','z','Intensity'])
MAX_RETURN_INTENSITY = 77.0
xyz_data[:, 3] /= MAX_RETURN_INTENSITY


print("removing the ground...")
offset = [0,0,0]       # optional: use to make calculations stable on large values
xyz_data_gr_2m,intensity_gr_2m = ground_removal.removeGround(xyz_data[:,:3],offset,returns=xyz_data[:,3],thresh=2.0,proc_path=output_dir)
xyz_data_gr_1m,intensity_gr_1m = ground_removal.removeGround(xyz_data[:,:3],offset,returns=xyz_data[:,3],thresh=1.0)
ground_pts = ground_removal.load_ground_surface(os.path.join(output_dir,'_ground_surface.ply'))



print("detecting trees...")
detector_addr = '/home/lloyd/Documents/projects/forest_3d_app/models/detection/hovermap2'
with open(os.path.join(detector_addr, 'raster_config.json')) as json_file:
    config_dict = json.load(json_file)
rasterTreeDetector = treeDetector.RasterDetector(**config_dict )
boxes = rasterTreeDetector.sliding_window(detector_addr,xyz_data_gr_2m,colour_data=intensity_gr_2m,ground_pts=ground_pts,
                                          windowSize = [22,22],stepSize = 15,overlap_thresh=3,returnBoxes=True)
labels = detection_tools.label_pcd_from_bbox(xyz_data_gr_1m, boxes, yxyx=True)


print("convert each tree to a separate array within a list...")
xyzr_list = []
for i in list(np.unique(labels)):
    if i > 0:
        xyzr_list.append(np.hstack((xyz_data_gr_1m[labels==i,:],intensity_gr_1m[labels==i])))


print("segmenting tree stems...")
segmenter_addr = '/home/lloyd/Documents/projects/forest_3d_app/models/stem_segmentation/hovermap'
with open(os.path.join(segmenter_addr, 'net_config.json')) as json_file:
    config_dict = json.load(json_file)
stem_model = stemSegmenter.VoxelStemSeg(**config_dict['model_dict'])
seg_list = stem_model.predict(xyzr_list,segmenter_addr,batchsize=20)


print("output segmented trees as individual pointclouds...")
for i,id in enumerate(list(np.unique(labels[labels>0]))):
    lidar_IO.writeXYZ_labelled(os.path.join(output_dir,'labelled_%i.asc'%(id)), seg_list[i][:,:3],labels=seg_list[i][:,3], delimiter=',')



print("outputting inventory...")
tree_tops = inventory_tools.get_tree_tops(xyz_data_gr_1m,labels)
heights = inventory_tools.get_tree_heights(tree_tops[:,:3],ground_pts)
inventory = np.hstack((tree_tops,heights[:,np.newaxis]))
utilities.write_csv(os.path.join(output_dir,'inventory.csv'),inventory,header='x,y,z,id,height')
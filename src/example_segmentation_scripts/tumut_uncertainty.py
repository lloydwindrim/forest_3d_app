'''
This script demonstrates the extraction of the certainty of the stem prediction. The outputs are:
- labelled_X.asc : an asc file for each tree, with columns 1,2,3 for x,y,z and columns 4,5 for certainty of label and label.
- stem_prob.asc : a single asc for all stem points of all trees (only stem points). Column 4 has the stem certainty. Useful
                    for comparing all stems on the same colour scale.
- inventory.csv : A single row for each tree, having the x,y,z of the tree-top, the filename, the mean stem certainty and
                    mean mean foliage certainty. Can be opened with excel.

All files can be loaded into cloudcompare.

'''

import sys
sys.path.insert(0, '../')
from forest3D import lidar_IO,stemSegmenter,inventory_tools,utilities

import os
import json
import numpy as np

output_dir = '/home/lloyd/Documents/projects/forest_3d_app/outputs'

print("reading the data into python...")
directory = '/home/lloyd/Documents/datasets/lidar/forestry_usyd/labelled_data/stem_segmentation/tumut/detected_pcds/'
xyzr_list,f_list = lidar_IO.readFromDirectory(directory,type='asc',output_fileList=True,delimiter=',',x=0,y=1,z=2,returns=3)


print("segmenting tree stems...")
segmenter_addr = '/home/lloyd/Documents/projects/forest_3d_app/models/stem_segmentation/tumut1'
with open(os.path.join(segmenter_addr, 'net_config.json')) as json_file:
    config_dict = json.load(json_file)
stem_model = stemSegmenter.VoxelStemSeg(**config_dict['model_dict'])
seg_list = stem_model.predict_uncertainty(xyzr_list,segmenter_addr)


print("outputting segmented trees with uncertainty...")
for i in range(len(seg_list)):
    lidar_IO.writeXYZ_labelled(os.path.join(output_dir,'labelled_%i.asc'%(i)), seg_list[i][:,:3],returns=seg_list[i][:,3],
                               labels=seg_list[i][:,4], delimiter=',')



print("outputting stem uncertainty pointcloud...")
for i in range(len(seg_list)):
    if i == 0:
        all_stems = seg_list[i][:,:4][seg_list[i][:,4]==2]
    else:
        all_stems = np.vstack(( all_stems , seg_list[i][:,:4][seg_list[i][:,4]==2] ))
lidar_IO.writeXYZ_labelled(os.path.join(output_dir,'stem_prob.asc'), all_stems[:,:3],labels=all_stems[:,3],delimiter=',')



print("computing inventory...")
inventory = np.zeros((len(seg_list),6),dtype='<U32')
for i in range(len(seg_list)):
    stem_prob = inventory_tools.get_seg_prob(probs=seg_list[i][:,3],labels=seg_list[i][:,4],target_class=2)
    foliage_prob = inventory_tools.get_seg_prob(probs=seg_list[i][:, 3], labels=seg_list[i][:, 4], target_class=1)
    tree_top = inventory_tools.get_single_tree_top(seg_list[i][:, :3])
    id = f_list[i].split('/')[-1]
    inventory[i,:] = np.hstack((tree_top,[id],[stem_prob],[foliage_prob]))
utilities.write_csv(os.path.join(output_dir,'inventory.csv'),inventory,header='x,y,z,id,stem_prob,foliage_prob')





import sys
sys.path.insert(0, '../')
from forest3D import lidar_IO,stemSegmenter

import os
import json

output_dir = '/home/lloyd/Documents/projects/forest_3d_app/outputs'

print("reading the data into python...")
directory = '/home/lloyd/Documents/datasets/lidar/forestry_usyd/labelled_data/stem_segmentation/tumut/detected_pcds/'
xyzr_list,f_list = lidar_IO.readFromDirectory(directory,type='asc',output_fileList=True,delimiter=',',x=0,y=1,z=2,returns=3)


print("segmenting tree stems...")
segmenter_addr = '/home/lloyd/Documents/projects/forest_3d_app/models/stem_segmentation/tumut2'
with open(os.path.join(segmenter_addr, 'net_config.json')) as json_file:
    config_dict = json.load(json_file)
stem_model = stemSegmenter.VoxelStemSeg(**config_dict['model_dict'])
seg_list = stem_model.predict(xyzr_list,segmenter_addr)


print("output trees as individual pointclouds...")
for i in range(len(seg_list)):
    lidar_IO.writeXYZ_labelled(os.path.join(output_dir,'labelled_%i.asc'%(i)), seg_list[i][:,:3],labels=seg_list[i][:,3], delimiter=',')


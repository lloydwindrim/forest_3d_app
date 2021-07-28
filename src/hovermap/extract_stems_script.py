#!/usr/bin/python

"""
extract_stems_script.py - Script to process pointclouds to pull out stems for high-res/hovermap data
"""

import os
from math import *
import random
import colorsys
import csv

import numpy as np
import laspy.file as lpf

import pickle

import treepointcloud
import tp_visualisation

import open3d as o3d

############################
# required paths to enter:


# Path to input data pointcloud (Uncomment and put in path to las file to process)
#input_path = 'example.las'

# Path to where processed data gets put (uncomment and specify a directory for processed output)
#proc_dir = 'processed_data_folder'


######################
# Script starts here
######################

# Create proc sub-directories if they don't exist
if not os.path.isdir(proc_dir):
    os.mkdir(proc_dir)
if not os.path.isdir(os.path.join(proc_dir,'trees')):
    os.mkdir(os.path.join(proc_dir,'trees'))
if not os.path.isdir(os.path.join(proc_dir,'csv')):
    os.mkdir(os.path.join(proc_dir,'csv'))

####################
# Load up las file
print("Loading las file ...")

las_file = lpf.File(input_path, mode = "r")
X = las_file.X*las_file.header.scale[0]+las_file.header.offset[0]
Y = las_file.Y*las_file.header.scale[1]+las_file.header.offset[1]
Z = las_file.Z*las_file.header.scale[2]+las_file.header.offset[2]

# Set "offset" to specify the "center" of the dataset, if required
offset = [0.0,0.0,0.0]

#############################
# Convert to pcd for open3D
xyz = np.zeros((X.shape[0], 3))
xyz[:, 0] = X-offset[0]
xyz[:, 1] = Y-offset[1]
xyz[:, 2] = Z-offset[2]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
X = None
Y = None
Z = None

######################################################################
# Use voxel downsampling to downsample pointcloud to 5cm resolution
print("Voxel down-sampling to 5cm ...")

downpcd = pcd.voxel_down_sample(voxel_size=0.05)

# Convert back into XYZ arrays
xyz = np.asarray(downpcd.points)
X = xyz[:,0]
Y = xyz[:,1]
Z = xyz[:,2]
downpcd = None

############################
# Ground plane segmentation

if True: # Perform ground segmentation
    print('Segmenting ground plane ...')
    ground = treepointcloud.GroundSurface(X,Y,Z)
    
    # Save out surface and ply to file
    print('Saving ground surface ...')
    output_ground_path = os.path.join(proc_dir,'ground_surface.ply')
    ground.export_ground_surface_ply(output_ground_path)
    
    output_ground_path_p = os.path.join(proc_dir,'ground_surface.p')
    pickle.dump( ground, open( output_ground_path_p, "wb" ) )
    
else: # OR, load from pickle:
    groundppath = os.path.join(proc_dir,'ground_surface.p')
    ground = pickle.load( open( groundppath, "rb" ) )

############################
# Find initial circle fits

if True: # Find circles in vertical layers
    print('Extracting circles from vertical layers ...')
    circles = treepointcloud.FindCircleStacks(X, Y, Z, dh=0.2, min_h=0.0, max_h=10.0)

    # export discovered circles to file
    outplypath = os.path.join(proc_dir,'extracted_circles_full.ply')
    tp_visualisation.Savecirclesply(outplypath, circles, offset=[0.0,0.0,0.0])

    # Pickle the circles for later analysis
    outppath = os.path.join(proc_dir,'extracted_circles_full.p')
    pickle.dump( circles, open( outppath, "wb" ) )
    
else: # OR load up pre-computed from pickle
    circleppath = os.path.join(proc_dir,'extracted_circles_full.p')
    circles = pickle.load( open( circleppath, "rb" ) )

#########################################################################
# Perform non-maxima suppression to remove excess false positive circles
keep_circle = np.array(len(circles)*[True])
circles2 = np.array(circles)
for i in range(circles2.shape[0]):
    print("\rNon-maxima suppression: %d/%d ..."%(i,circles2.shape[0]),end='')
    if not keep_circle[i]:
        continue
    inds1 = np.nonzero(circles2[:,3] == circles2[i,3])[0]
    rel_points = circles2[inds1,0:2]-circles2[i,0:2]
    r_all = np.linalg.norm(rel_points,axis=1)
    sumr = circles2[inds1,2]+circles2[i,2]
    inds_same = inds1[np.nonzero(r_all < sumr)[0]]
    best_i = inds_same[np.argmax(circles2[inds_same,4])]
    keep_circle[inds_same] = False
    keep_circle[best_i] = True

inds_keep = np.nonzero(keep_circle)[0]

#circle_update = list(circles2[inds_keep,:])
#outplypath = os.path.join(proc_dir,'extracted_circles_full2.ply')
#tp_visualisation.Savecirclesply(outplypath, circle_update, offset=[0.0,0.0,0.0])

#########################################################################
# Form stems from circles using a stem climbing approach
print('Constructing Stem Segments from circular sections ...')
grid_circle = treepointcloud.GridCircles(circles2[inds_keep,:])
stem_data = []
for ix in range(grid_circle.Nx):
    for iy in range(grid_circle.Ny):
        print("\rScanning Grid %d/%d, stems: %d ..."%(ix,iy,len(stem_data)),end='')
        circles_now = grid_circle.data[ix][iy]
        if len(circles_now) < 10:
            continue
        tagged = -1*np.ones(len(circles_now))
        for i in range(circles_now.shape[0]):
            if tagged[i] >= 0 or circles_now[i,3] == circles_now[-1,3]:
                continue
            stack = [i]
            nexti = i
            while True:
                inds1 = np.nonzero(np.logical_and(circles_now[:,3] <= circles_now[nexti,3]+2.5, circles_now[:,3]>circles_now[nexti,3]))[0]
                if len(inds1) == 0:
                    break
                rel_points = circles_now[inds1,0:2]-circles_now[nexti,0:2]
                r_all = np.linalg.norm(rel_points,axis=1)
                inds2 = np.nonzero(np.logical_and(r_all < 0.1,np.fabs(circles_now[inds1,2]-circles_now[nexti,2])<0.2*circles_now[nexti,2]))[0]
                if len(inds2) == 0:
                    break
                inds = inds1[inds2]
                minz = np.min(circles_now[inds,3]-circles_now[nexti,3])+circles_now[nexti,3]
                inds3 = inds[np.nonzero(circles_now[inds,3] <= (minz+0.01))[0]]
                ibest = inds3[np.argmin(np.fabs(circles_now[inds3,2]-circles_now[nexti,2]))]
                stack.append(ibest)
                nexti = ibest
                tagged[ibest] = 1
            if len(stack) > 10:
                #stem_data.append(circles_now[stack,:])
                stem = []
                for s in stack:
                    stem.append(list(circles_now[s,:]))
                stem_data.append(stem)


##################################
# Merge stems based on overlap

print('Merging into trees ...')
stem_data_final = treepointcloud.MergetoTrees(stem_data)

# Get list of tree locations
avg_xy = []
for tree in stem_data_final:
    mx = np.mean([d[0] for d in tree])
    my = np.mean([d[1] for d in tree])
    if len(avg_xy) == 0:
        avg_xy.append([mx,my])
    else:
        dists = [sqrt(pow(mx-d[0],2)+pow(my-d[1],2)) for d in avg_xy]
        if np.min(dists) > 2.0:
            avg_xy.append([mx,my])

# save locs to csv 
output_csv = os.path.join(proc_dir,'tree_locs.csv')
fcsv = open(output_csv, "w");
writer = csv.writer(fcsv, delimiter=',')
#writer.writerow(['X','Y'])
for loc in avg_xy:
    writer.writerow([loc[0],loc[1]])
fcsv.close()

############################################################
# pull out points for a given tree using stem as reference

print('Extracting tree/stem points for each found tree ...')
tree_points = []
stem_points = []
c = 0
for tree in stem_data_final:
    print('finding points for tree %d'%(c))
    (stem_p, tree_p) = treepointcloud.ExtractPointsusingStem(tree,X,Y,Z,dh=0.5,r_thresh_out=0.05,r_thresh_in=0.05,tree_size=3.0)
    tree_points.append(tree_p)
    stem_points.append(stem_p)
    c += 1


################################################
# Exporting all extracted data to various files
################################################

print('Exporting files for visualisation ...')

###############################
# export kept stems to file (all stems in one ply file)
outplypath = os.path.join(proc_dir,'stems_full.ply')
tp_visualisation.Savestemply_all(outplypath, stem_data_final, offset=[0.0,0.0,0.0])

###########################
# Save tree-by-tree stems
outpath_dir = os.path.join(proc_dir,'trees')
num = 0
for stem in stem_data_final:
    outplypath_pointsstem = os.path.join(outpath_dir,'stemsections_%03d.ply'%(num))
    tp_visualisation.Savestemply(outplypath_pointsstem, stem, offset=[0.0,0.0,0.0])
    num += 1

######################################
# Output list of tree rings as csv
output_csv = os.path.join(proc_dir,'stem_rings.csv')
fcsv = open(output_csv, "w");
writer = csv.writer(fcsv, delimiter=',')
writer.writerow(['tree_id','X','Y','Z','DBH_cm'])
tree_id = 0
for stem in stem_data_final:
    for circle in stem:
        writer.writerow([tree_id,circle[0],-circle[1],-circle[3],2*100*circle[2]])
    tree_id += 1
fcsv.close()

################################
# Export segmented pointclouds

# Save to one big file
outplypath_points = os.path.join(proc_dir,'tree_points.ply')
tp_visualisation.Savetreepoints_all(outplypath_points, tree_points, offset=[0.0,0.0,0.0])

outplypath_points2 = os.path.join(proc_dir,'stem_points.ply')
tp_visualisation.Savestempoints_all(outplypath_points2, stem_points, offset=[0.0,0.0,0.0])

# Save tree-by-tree files
outpath_dir = os.path.join(proc_dir,'trees')
num = 0
for tree in tree_points:
    outplypath_pointstree = os.path.join(outpath_dir,'treepoints_%03d.ply'%(num))
    tp_visualisation.Savetreepoints(outplypath_pointstree, tree, offset=[0.0,0.0,0.0])
    num += 1
num = 0
for stem in stem_points:
    outplypath_pointsstem = os.path.join(outpath_dir,'stempoints_%03d.ply'%(num))
    tp_visualisation.Savestempoints(outplypath_pointsstem, stem, offset=[0.0,0.0,0.0])
    outcsvpath_pointsstem = os.path.join(proc_dir,'csv','stempoints_%03d.csv'%(num))
    tp_visualisation.Savestempoints_csv(outcsvpath_pointsstem, stem)
    num += 1



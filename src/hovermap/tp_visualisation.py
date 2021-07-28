#!/usr/bin/python

"""
tp_visualisation.py: Functions for exporting processed data from treepointcloud 
for visualisation
"""

import os
from math import *
import random
import colorsys
import csv

import numpy as np

###################################################
# Functions for exporting models for visualisation

# Savecirclesply: saves a ply file containing extracted circular sections
def Savecirclesply(filepath, circle_data, offset=None, R=None):
    
    Ndotsl = 15
    
    if offset is None:
        xm = 0.0
        ym = 0.0
        zm = 0.0
    else:
        xm = offset[0]
        ym = offset[1]
        zm = offset[2]
    
    c_good = len(circle_data)
    
    f = open(filepath, "w");
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write('comment %.3f\n'%(xm))
    f.write('comment %.3f\n'%(ym))
    f.write('comment %.3f\n'%(zm))
    f.write('element vertex %d\n'%(c_good*Ndotsl))
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    f.write('element edge %d\n'%(c_good*Ndotsl))
    f.write('property int vertex1\n')
    f.write('property int vertex2\n')
    f.write('end_header\n')
    t = np.linspace(0,2*pi,Ndotsl)
    for circle in circle_data:
        x = circle[2]*np.sin(t)+circle[0]
        y = circle[2]*np.cos(t)+circle[1]
        z = np.array([circle[3] for i in range(len(t))])
        if not R is None:
            stem_points_trans = np.matmul(R,np.array([x,y,z])).T
            x = stem_points_trans[:,0]
            y = stem_points_trans[:,1]
            z = stem_points_trans[:,2]
        for i in range(x.shape[0]):
            f.write('%.4f %.4f %.4f\n'%(x[i]-xm,y[i]-ym,z[i]-zm))
    c_stem = 0
    for circle in circle_data:
        for i in range(t.shape[0]-1):
            f.write('%d %d\n'%(Ndotsl*c_stem+i,Ndotsl*c_stem+i+1))
        f.write('%d %d\n'%(Ndotsl*c_stem+t.shape[0]-1,Ndotsl*c_stem))
        c_stem += 1
    f.close()

# Savestemply_all: saves a ply file containing all extracted stems
def Savestemply_all(filepath, stem_data, offset=None, R=None):
    
    Ndotsl = 15
    
    if offset is None:
        xm = 0.0
        ym = 0.0
        zm = 0.0
    else:
        xm = offset[0]
        ym = offset[1]
        zm = offset[2]
    
    c_good = 0
    nface = 0
    for stem in stem_data:
        c_good += len(stem)
        nface += (len(stem)-1)*(Ndotsl)
    #c_good = len(stem_data)
    
    f = open(filepath, "w");
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write('comment %.3f\n'%(xm))
    f.write('comment %.3f\n'%(ym))
    f.write('comment %.3f\n'%(zm))
    f.write('element vertex %d\n'%(c_good*Ndotsl))
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    f.write('element edge %d\n'%(c_good*Ndotsl))
    f.write('property int vertex1\n')
    f.write('property int vertex2\n')
    f.write('element face %d\n'%(nface))
    f.write('property list uchar int vertex_indices\n')
    f.write('end_header\n')
    t = np.linspace(0,2*pi,Ndotsl)
    for stem in stem_data:
        for circle in stem:
            x = circle[2]*np.sin(t)+circle[0]
            y = circle[2]*np.cos(t)+circle[1]
            z = np.array([circle[3] for i in range(len(t))])
            if not R is None:
                stem_points_trans = np.matmul(R,np.array([x,y,z])).T
                x = stem_points_trans[:,0]
                y = stem_points_trans[:,1]
                z = stem_points_trans[:,2]
            for i in range(x.shape[0]):
                f.write('%.4f %.4f %.4f\n'%(x[i]-xm,y[i]-ym,z[i]-zm))
    c_stem = 0
    for stem in stem_data:
        for circle in stem:
            for i in range(t.shape[0]-1):
                f.write('%d %d\n'%(Ndotsl*c_stem+i,Ndotsl*c_stem+i+1))
            f.write('%d %d\n'%(Ndotsl*c_stem+t.shape[0]-1,Ndotsl*c_stem))
            c_stem += 1
    c_stem = 0
    for k in range(len(stem_data)):
        stem = stem_data[k]
        for j in range(len(stem)-1):
            circle = stem[j]
            for i in range(t.shape[0]-1):
                f.write('4 %d %d %d %d\n'%(Ndotsl*c_stem+i+1,Ndotsl*c_stem+i,Ndotsl*(c_stem+1)+i,Ndotsl*(c_stem+1)+i+1))
            f.write('4 %d %d %d %d\n'%(Ndotsl*c_stem,Ndotsl*c_stem+t.shape[0]-1,Ndotsl*(c_stem+1)+t.shape[0]-1,Ndotsl*(c_stem+1)))
            c_stem += 1
        c_stem += 1
    f.close()

# Savestemply: saves a ply file containing extracted stems
def Savestemply(filepath, stem, offset=None, R=None):
    
    Ndotsl = 15
    
    if offset is None:
        xm = 0.0
        ym = 0.0
        zm = 0.0
    else:
        xm = offset[0]
        ym = offset[1]
        zm = offset[2]
    
    c_good = len(stem)
    nface = (len(stem)-1)*Ndotsl
    
    f = open(filepath, "w");
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write('comment %.3f\n'%(xm))
    f.write('comment %.3f\n'%(ym))
    f.write('comment %.3f\n'%(zm))
    f.write('element vertex %d\n'%(c_good*Ndotsl))
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    f.write('element edge %d\n'%(c_good*Ndotsl))
    f.write('property int vertex1\n')
    f.write('property int vertex2\n')
    f.write('element face %d\n'%(nface))
    f.write('property list uchar int vertex_indices\n')
    f.write('end_header\n')
    t = np.linspace(0,2*pi,Ndotsl)
    for circle in stem:
        x = circle[2]*np.sin(t)+circle[0]
        y = circle[2]*np.cos(t)+circle[1]
        z = np.array([circle[3] for i in range(len(t))])
        if not R is None:
            stem_points_trans = np.matmul(R,np.array([x,y,z])).T
            x = stem_points_trans[:,0]
            y = stem_points_trans[:,1]
            z = stem_points_trans[:,2]
        for i in range(x.shape[0]):
            f.write('%.4f %.4f %.4f\n'%(x[i]-xm,y[i]-ym,z[i]-zm))
    c_stem = 0
    for circle in stem:
        for i in range(t.shape[0]-1):
            f.write('%d %d\n'%(Ndotsl*c_stem+i,Ndotsl*c_stem+i+1))
        f.write('%d %d\n'%(Ndotsl*c_stem+t.shape[0]-1,Ndotsl*c_stem))
        c_stem += 1
    c_stem = 0
    for j in range(len(stem)-1):
        circle = stem[j]
        for i in range(t.shape[0]-1):
            f.write('4 %d %d %d %d\n'%(Ndotsl*c_stem+i+1,Ndotsl*c_stem+i,Ndotsl*(c_stem+1)+i,Ndotsl*(c_stem+1)+i+1))
        f.write('4 %d %d %d %d\n'%(Ndotsl*c_stem,Ndotsl*c_stem+t.shape[0]-1,Ndotsl*(c_stem+1)+t.shape[0]-1,Ndotsl*(c_stem+1)))
        c_stem += 1
    f.close()

# Savestempoints_all: saves a ply file containing extracted stem points for all stems
def Savestempoints_all(filepath, stem_points, offset=None, R=None):
    
    if offset is None:
        xm = 0.0
        ym = 0.0
        zm = 0.0
    else:
        xm = offset[0]
        ym = offset[1]
        zm = offset[2]
    
    nump = 0 # count total points
    for stem in stem_points:
        for section in stem:
            nump += len(section[0])
    
    f = open(filepath, "w");
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write('comment %.3f\n'%(xm))
    f.write('comment %.3f\n'%(ym))
    f.write('comment %.3f\n'%(zm))
    f.write('element vertex %d\n'%(nump))
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    f.write('property uchar diffuse_red\n')
    f.write('property uchar diffuse_green\n')
    f.write('property uchar diffuse_blue\n')
    f.write('end_header\n')
    c = 0
    for stem in stem_points:
        for section in stem:
            X = section[0]
            Y = section[1]
            Z = section[2]
            (R,G,B) = colorsys.hsv_to_rgb(random.random(), 1.0, 1.0)
            #R = 1.0
            #G = 0
            #B = 0
            for i in range(len(X)):
                f.write('%.4f %.4f %.4f %d %d %d\n'%(X[i]-xm,Y[i]-ym,Z[i]-zm,int(255*R),int(255*G),int(255*B)))
                c += 1
    f.close()

# Savestempoints: saves a ply file containing extracted stem points for a stem
def Savestempoints(filepath, stem, offset=None, R=None):
    
    if offset is None:
        xm = 0.0
        ym = 0.0
        zm = 0.0
    else:
        xm = offset[0]
        ym = offset[1]
        zm = offset[2]
    
    nump = 0 # count total points
    for section in stem:
        nump += len(section[0])
    
    f = open(filepath, "w");
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write('comment %.3f\n'%(xm))
    f.write('comment %.3f\n'%(ym))
    f.write('comment %.3f\n'%(zm))
    f.write('element vertex %d\n'%(nump))
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    f.write('property uchar diffuse_red\n')
    f.write('property uchar diffuse_green\n')
    f.write('property uchar diffuse_blue\n')
    f.write('end_header\n')
    c = 0
    for section in stem:
        X = section[0]
        Y = section[1]
        Z = section[2]
        #(R,G,B) = colorsys.hsv_to_rgb(random.random(), 1.0, 1.0)
        R = 1.0
        G = 0
        B = 0
        for i in range(len(X)):
            f.write('%.4f %.4f %.4f %d %d %d\n'%(X[i]-xm,Y[i]-ym,Z[i]-zm,int(255*R),int(255*G),int(255*B)))
            c += 1
    f.close()

# Savestempoints_csv: saves a csv file containing extracted stem points for a stem
def Savestempoints_csv(filepath, stem):
    
    nump = 0 # count total points
    for section in stem:
        nump += len(section[0])
    
    fcsv = open(filepath, "w");
    writer = csv.writer(fcsv, delimiter=',')
    writer.writerow(['X','Y','Z','R','G','B'])  
    c = 0
    for section in stem:
        X = section[0]
        Y = section[1]
        Z = section[2]
        (R,G,B) = colorsys.hsv_to_rgb(random.random(), 1.0, 1.0)
        #R = 1.0
        #G = 0
        #B = 0
        for i in range(len(X)):
            writer.writerow([X[i],Z[i],Y[i],int(255*R),int(255*G),int(255*B)])
            c += 1
    fcsv.close()

# Savetreepoints_all: saves a ply file containing all extracted tree points
def Savetreepoints_all(filepath, tree_points, offset=None, R=None):
    
    if offset is None:
        xm = 0.0
        ym = 0.0
        zm = 0.0
    else:
        xm = offset[0]
        ym = offset[1]
        zm = offset[2]
    
    nump = 0 # count total points
    for tree in tree_points:
        nump += len(tree[0])
    
    f = open(filepath, "w");
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write('comment %.3f\n'%(xm))
    f.write('comment %.3f\n'%(ym))
    f.write('comment %.3f\n'%(zm))
    f.write('element vertex %d\n'%(nump))
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    f.write('property uchar diffuse_red\n')
    f.write('property uchar diffuse_green\n')
    f.write('property uchar diffuse_blue\n')
    f.write('end_header\n')
    c = 0
    for tree in tree_points:
        X = tree[0]
        Y = tree[1]
        Z = tree[2]
        (R,G,B) = colorsys.hsv_to_rgb(random.random(), 1.0, 1.0)
        for i in range(len(X)):
            f.write('%.4f %.4f %.4f %d %d %d\n'%(X[i]-xm,Y[i]-ym,Z[i]-zm,int(255*R),int(255*G),int(255*B)))
            c += 1
    f.close()

# Savetreepoints: saves a ply file containing extracted points for a single tree
def Savetreepoints(filepath, tree, offset=None, R=None):
    
    if offset is None:
        xm = 0.0
        ym = 0.0
        zm = 0.0
    else:
        xm = offset[0]
        ym = offset[1]
        zm = offset[2]
    
    nump = len(tree[0])
    
    f = open(filepath, "w");
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write('comment %.3f\n'%(xm))
    f.write('comment %.3f\n'%(ym))
    f.write('comment %.3f\n'%(zm))
    f.write('element vertex %d\n'%(nump))
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    f.write('property uchar diffuse_red\n')
    f.write('property uchar diffuse_green\n')
    f.write('property uchar diffuse_blue\n')
    f.write('end_header\n')
    c = 0
    X = tree[0]
    Y = tree[1]
    Z = tree[2]
    (R,G,B) = colorsys.hsv_to_rgb(random.random(), 1.0, 1.0)
    for i in range(len(X)):
        f.write('%.4f %.4f %.4f %d %d %d\n'%(X[i]-xm,Y[i]-ym,Z[i]-zm,int(255*R),int(255*G),int(255*B)))
        c += 1
    f.close()






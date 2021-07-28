#!/usr/bin/python

"""
treepointcloud.py: Functions for segmenting high density TLS pointclouds into tree
level information
"""

import os
from math import *
import random
import csv

import numpy as np
import laspy.file as lpf

from scipy import optimize
from scipy.spatial import Delaunay

###################################################################
# GroundSurface: used to extract ground from local minima points

class GroundSurface(object):
    def __init__(self,X,Y,Z,dx=2.0):
        self.dx = dx
        self.xmin = np.min(X)
        self.ymin = np.min(Y)
        self.xmax = np.max(X)+0.1
        self.ymax = np.max(Y)+0.1
        self.zmax = np.max(Z)
        self.Nx = int(ceil((self.xmax-self.xmin)/self.dx))
        self.Ny = int(ceil((self.ymax-self.ymin)/self.dx))
        
        self.Z = Z
        self.inds = np.floor((X-self.xmin)/self.dx) + self.Nx*np.floor((Y-self.ymin)/self.dx)
        self.inds = self.inds.astype(int)
        self.ground_grid = self.zmax*np.ones(self.Nx*self.Ny)
        
        for i in range(len(self.Z)):
            self.check_point(i)
        
        self.Z = None
        self.inds = None
    
    def check_point(self,i):
        if self.Z[i] < self.ground_grid[self.inds[i]]:
            self.ground_grid[self.inds[i]] = self.Z[i]
    
    def median_filter_thresh(self,h_thresh=5.0):
        for ix in range(self.Nx):
            for iy in range(self.Ny):
                if self.ground_grid[iy*self.Nx+ix] >= self.zmax:
                    continue
                z = self.ground_grid[iy*self.Nx+ix]
                zs = []
                for dx in [-1,0,1]:
                    for dy in [-1,0,1]:
                        if dx == 0 and dy == 0:
                            continue
                        ix2 = ix+dx
                        iy2 = iy+dy
                        if ix2 >= 0 and ix2 < self.Nx and iy2 >= 0 and iy2 < self.Ny and self.ground_grid[iy2*self.Nx+ix2] < self.zmax:
                            zs.append(self.ground_grid[iy2*self.Nx+ix2])
                zm = np.median(zs)
                if fabs(z-zm) > h_thresh:
                    self.ground_grid[iy*self.Nx+ix] = zm
    
    def export_ground_surface_ply(self, outputpath):
    
        offset = [0.0,0.0,0.0]
    
        # convert grid to set of points
        ground_points = []
        for i in range(self.Nx*self.Ny):
            if self.ground_grid[i] < self.zmax:
                ground_points.append([self.dx*float(i%self.Nx)+self.xmin,self.dx*int(i/self.Nx)+self.ymin,self.ground_grid[i]])
        ground_points = np.array(ground_points)
    
        # Build Delaunay triangulation
        tri = Delaunay(ground_points[:,0:2])

        # Save output surface to ply file
        f = open(outputpath, "w");
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('comment %.3f\n'%(offset[0]))
        f.write('comment %.3f\n'%(offset[1]))
        f.write('comment %.3f\n'%(offset[2]))
        f.write('element vertex %d\n'%(ground_points.shape[0]))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('element face %d\n'%(tri.simplices.shape[0]))
        f.write('property list uchar int vertex_indices\n')
        f.write('end_header\n')
        for i in range(ground_points.shape[0]):
            f.write('%.4f %.4f %.4f\n'%(ground_points[i,0],ground_points[i,1],ground_points[i,2]))
        for i in range(tri.simplices.shape[0]):
            f.write('3 %d %d %d\n'%(tri.simplices[i,0],tri.simplices[i,1],tri.simplices[i,2]))
        f.close()

#######################################
# Class to break points into grids

class GridPoints(object):
    def __init__(self,Xs,Ys,Zs,dx=2.0):
        dx2 = 0.5*dx
        self.xmin = np.min(Xs)
        self.ymin = np.min(Ys)
        self.xmax = np.max(Xs)
        self.ymax = np.max(Ys)
        self.Nx = int(ceil((self.xmax-self.xmin)/dx2))-1
        self.Ny = int(ceil((self.ymax-self.ymin)/dx2))-1
        self.data = []
        for ix in range(self.Nx):
            self.data.append([])
            for iy in range(self.Ny):
                xminc = self.xmin+ix*dx2
                yminc = self.ymin+iy*dx2
                xmaxc = self.xmin+ix*dx2+dx
                ymaxc = self.ymin+iy*dx2+dx
                inds = np.nonzero(np.logical_and(Xs >= xminc, np.logical_and(Xs < xmaxc, np.logical_and(Ys >= yminc, Ys < ymaxc))))
                #self.data[-1].append(np.concatenate((Xs[inds],Ys[inds],Zs[inds]),axis=0))
                self.data[-1].append(np.stack((Xs[inds],Ys[inds],Zs[inds]),axis=1))

# GridCircles: sort extracted circles into grids
class GridCircles(object):
    def __init__(self,circles,dx=5.0):
        dx2 = 0.5*dx
        self.xmin = np.min(circles[:,0])
        self.ymin = np.min(circles[:,1])
        self.xmax = np.max(circles[:,0])
        self.ymax = np.max(circles[:,1])
        self.Nx = int(ceil((self.xmax-self.xmin)/dx2))-1
        self.Ny = int(ceil((self.ymax-self.ymin)/dx2))-1
        self.data = []
        for ix in range(self.Nx):
            self.data.append([])
            for iy in range(self.Ny):
                xminc = self.xmin+ix*dx2
                yminc = self.ymin+iy*dx2
                xmaxc = self.xmin+ix*dx2+dx
                ymaxc = self.ymin+iy*dx2+dx
                inds = np.nonzero(np.logical_and(circles[:,0] >= xminc, np.logical_and(circles[:,0] < xmaxc, np.logical_and(circles[:,1] >= yminc, circles[:,1] < ymaxc))))[0]
                #self.data[-1].append(circles[inds,:])
                data = np.concatenate((circles[inds,:],np.array([inds]).T),axis=1) # add ind to last column
                self.data[-1].append(data)


#############################################################
# functions for fitting a circle, used by optimize.leastsq

def fitfunc_circle(p,x):
    r = np.linalg.norm(x-np.array([p[0],p[1]]),axis=1)
    return r

def errfunc_circle(p,x):
    e = p[2] - fitfunc_circle(p, x)
    return e

###################################################################################
# FindCircleBest: only returns the one best candidate, if one fits the thresholds
def FindCircleBest(points):
    
    Ntest = 200
    r_min = 0.075
    r_max = 0.5
    r_thresh_out = 0.05
    r_thresh_in = 0.05
    r_thresh_in2 = 0.1
    Ninner_thresh = 20
    Ngood_min = 200
    
    Nbest = 0
    best_inliers = []
    best_inds = []
    
    if points.shape[0] < 20: # bail on low number of points
        return None
    
    # cycle through tests
    for ti in range(Ntest):
            
        # grab random sample of three points
        #inds = np.random.permutation(points.shape[0])[0:3]
        inds = [np.random.randint(0,points.shape[0]),np.random.randint(0,points.shape[0]),np.random.randint(0,points.shape[0])]

        # perform circle fitting on points (circumcenter of three points)
        a = points[inds[0],0:2]
        b = points[inds[1],0:2]
        c = points[inds[2],0:2]
        d = 2*(a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))
        if d == 0:
            continue
        xcent = (1.0/d)*((a[0]*a[0] + a[1]*a[1])*(b[1]-c[1]) + (b[0]*b[0] + b[1]*b[1])*(c[1]-a[1]) + (c[0]*c[0] + c[1]*c[1])*(a[1]-b[1]))
        ycent = (1.0/d)*((a[0]*a[0] + a[1]*a[1])*(c[0]-b[0]) + (b[0]*b[0] + b[1]*b[1])*(a[0]-c[0]) + (c[0]*c[0] + c[1]*c[1])*(b[0]-a[0]))
        r = np.mean(np.linalg.norm(points[inds,0:2]-np.array([xcent,ycent]),axis=1))

        if r < r_min or r > r_max: # abort if not a good fit
            continue

        # determine residuals for remaining points
        rel_points = points[:,0:2]-np.array([xcent,ycent])
        r_all = np.linalg.norm(rel_points,axis=1)
        r_diff = r_all - r
        good_ind = np.nonzero(np.logical_and(r_diff < r_thresh_out, r_diff > -r_thresh_in))[0]
        Ngood = len(good_ind)
        Ninner = len(np.nonzero(r_diff < -r_thresh_in2)[0])
        
        # Look for circumferential coverage
        theta = np.arctan2(rel_points[good_ind,1],rel_points[good_ind,0])
        (counts,_) = np.histogram(theta, bins=np.linspace(-pi,pi,12))
        if not np.all(counts > 1):
            continue
        
        # check if it's good enough and better than previous best
        if Ninner <= Ninner_thresh and Ngood >= Ngood_min and Ngood > Nbest:
            p0 = [xcent,ycent,r]
            # re-fit parameters with inliers
            p1, success = optimize.leastsq(errfunc_circle, p0[:], args=(points[good_ind,0:2]))
            if success and p1[2] > r_min and p1[2] < r_max:
                Nbest = Ngood
                params = [p1[0],p1[1],p1[2],Ngood,Ninner]
                best_inliers = good_ind
                best_inds = inds
                
    # return found stem
    if Nbest > 0:
        return params
    else:
        return None

#########################################################################################
# FindCircles: can find multiple circles that fit a threshold, non-maxima suppression of
# weaker overlapping candidates
def FindCircles(points):
    
    Ntest = 200
    r_min = 0.075
    r_max = 0.5
    r_thresh_out = 0.05
    r_thresh_in = 0.05
    r_thresh_in2 = 0.05 # was 0.1
    Ninner_thresh = 5
    #Ngood_min = 200
    Ngood_min = 50 # brought down a bit for 5cm processing
    
    params = np.zeros((Ntest, 4)) # each entry: [x,y,r,N]
    stem_inds = []
    Nfound = 0
    
    if points.shape[0] < 20: # bail on low number of points
        return None
    
    # cycle through tests
    for ti in range(Ntest):
            
        # grab random sample of three points
        #inds = np.random.permutation(points.shape[0])[0:3] # this is slow!
        inds = [np.random.randint(0,points.shape[0]),np.random.randint(0,points.shape[0]),np.random.randint(0,points.shape[0])]

        # perform circle fitting on points (circumcenter of three points)
        a = points[inds[0],0:2]
        b = points[inds[1],0:2]
        c = points[inds[2],0:2]
        d = 2*(a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))
        if d == 0:
            continue
        xcent = (1.0/d)*((a[0]*a[0] + a[1]*a[1])*(b[1]-c[1]) + (b[0]*b[0] + b[1]*b[1])*(c[1]-a[1]) + (c[0]*c[0] + c[1]*c[1])*(a[1]-b[1]))
        ycent = (1.0/d)*((a[0]*a[0] + a[1]*a[1])*(c[0]-b[0]) + (b[0]*b[0] + b[1]*b[1])*(a[0]-c[0]) + (c[0]*c[0] + c[1]*c[1])*(b[0]-a[0]))
        r = np.mean(np.linalg.norm(points[inds,0:2]-np.array([xcent,ycent]),axis=1))

        if r < r_min or r > r_max: # abort if not a good fit
            continue

        # determine residuals for remaining points
        rel_points = points[:,0:2]-np.array([xcent,ycent])
        r_all = np.linalg.norm(rel_points,axis=1)
        r_diff = r_all - r
        good_ind = np.nonzero(np.logical_and(r_diff < r_thresh_out, r_diff > -r_thresh_in))[0]
        Ngood = len(good_ind)
        Ninner = len(np.nonzero(r_diff < -r_thresh_in2)[0])
        
        # Look for circumferential coverage
        theta = np.arctan2(rel_points[good_ind,1],rel_points[good_ind,0])
        (counts,_) = np.histogram(theta, bins=np.linspace(-pi,pi,12))
        #if not np.all(counts > 1):
        if np.sum(counts > 1) < 6: # was 9
            continue
        
        # check if candidate is good enough to be checked at all
        if Ninner <= Ninner_thresh and Ngood >= Ngood_min:
            
            # check if it fits inside any existing candidates
            if Nfound > 0:
                rel_points = params[:,0:2]-np.array([xcent,ycent])
                r_all = np.linalg.norm(rel_points,axis=1)
                sumr = params[:,2]+np.array([r])
                inds_same = np.nonzero(r_all < sumr)[0]
                if len(inds_same) == 0: # new circle
                    #p0 = [xcent,ycent,r]
                    #p1, success = optimize.leastsq(errfunc_circle, p0[:], args=(points[good_ind,0:2]))
                    p1 = [xcent,ycent,r]
                    success = True
                    if success and p1[2] > r_min and p1[2] < r_max:
                        params[Nfound,:] = [p1[0],p1[1],p1[2],Ngood]
                        stem_inds.append(good_ind)
                        Nfound += 1
                elif len(inds_same) > 1: # shouldn't happen for sensible circles
                    #print('bogus!')
                    continue
                else:
                    if Ngood > params[inds_same[0],3]: # better fit than previous overlapping one
                        #p0 = [xcent,ycent,r]
                        #p1, success = optimize.leastsq(errfunc_circle, p0[:], args=(points[good_ind,0:2]))
                        p1 = [xcent,ycent,r]
                        success = True
                        if success and p1[2] > r_min and p1[2] < r_max:
                            params[inds_same[0],:] = [p1[0],p1[1],p1[2],Ngood]
                            stem_inds[inds_same[0]] = good_ind
                    else: # not better than existing, ignore
                        continue
                
            else: # first candidate
                #p0 = [xcent,ycent,r]
                #p1, success = optimize.leastsq(errfunc_circle, p0[:], args=(points[good_ind,0:2]))
                p1 = [xcent,ycent,r]
                success = True
                if success and p1[2] > r_min and p1[2] < r_max:
                    params[Nfound,:] = [p1[0],p1[1],p1[2],Ngood]
                    stem_inds.append(good_ind)
                    Nfound += 1
    
    # return found stem
    if Nfound > 0:
        return params[0:Nfound,:]
    else:
        return None

##################################################################################
# FindCirclesStacks: breaks pointcloud up into vertical slices and searches for 
# circles in each layer
#
def FindCircleStacks(X, Y, Z, dh=0.5, min_h=None, max_h=None):
    
    if min_h == None:
        min_h = np.min(Z)
    if max_h == None:
        max_h = np.max(Z)-dh
    
    N = int((max_h-min_h)/dh)
    circles = []
    for i in range(N):
        print("\rScanning Slice %d/%d, (gridding)..."%(i+1,N),end='')
        minz = min_h+i*dh
        maxz = min_h+(i+1)*dh
        inds = np.nonzero(np.logical_and(Z > minz,Z < maxz))
        Xs = X[inds]
        Ys = Y[inds]
        Zs = Z[inds]
        grid = GridPoints(Xs,Ys,Zs,dx=2.0)
        num_g = 0
        for ix in range(grid.Nx):
            for iy in range(grid.Ny):
                print("\rScanning Slice %d/%d, Grid %d/%d ..."%(i+1,N,num_g+1,grid.Nx*grid.Ny),end='')
                params = FindCircles(grid.data[ix][iy])
                if not params is None:
                    for j in range(params.shape[0]):
                        circles.append([params[j,0],params[j,1],params[j,2],minz+dh/2,params[j,3]])
                num_g += 1
    
    return circles


#############################################################
# functions for fitting a stem from stacks of circles via
# line fitting through the circle centers

def FindStemsfromCircles(circles):
    
    Ntest = 100
    r_min = 0.075
    r_max = 0.5
    Ngood_min = 5
    Nbest = 0
    #z_range_thresh = 10.0
    z_range_thresh = 2.0
    
    params = np.zeros((Ntest, 4)) # each entry: [x,y,r,N]
    circle_inds = []
    Nfound = 0
    
    if circles.shape[0] < Ngood_min: # bail on low number of circles
        return None
    
    # cycle through tests
    for ti in range(Ntest):
            
        # grab random sample of two circles
        inds = np.random.permutation(circles.shape[0])[0:2]
        
        # perform line fitting on points
        if circles[inds[0],3] == circles[inds[1],3]:
            continue
        p1 = circles[inds[0],[0,1,3]]
        rel = circles[inds[1],[0,1,3]]-circles[inds[0],[0,1,3]]
        u1 = rel/np.linalg.norm(rel)
        
        # compare distance to line for remaining circles in set
        xc = p1[0]+(u1[0]/u1[2])*(circles[:,3]-p1[2])
        dx = circles[:,0]-xc
        yc = p1[1]+(u1[1]/u1[2])*(circles[:,3]-p1[2])
        dy = circles[:,1]-yc
        rel = np.stack((dx,dy),axis=1)
        r = np.linalg.norm(rel,axis=1)
        good_ind = np.nonzero(r < 0.1)[0]
        Ngood = len(good_ind)
        
        # Calculate tests for vertical extent
        zmin = np.min(circles[good_ind,3])
        zmax = np.max(circles[good_ind,3])
        z_range = zmax-zmin
        #nbins = 8
        nbins = ceil(z_range/0.5)
        (z_counts,_) = np.histogram(circles[good_ind,3], bins=np.linspace(zmin-0.01,zmax+0.01,nbins))
            
        # check for good stem found
        if Ngood >= Ngood_min and z_range >= z_range_thresh and np.all(z_counts > 1):
            
            # for now just keep best, fix up later
            if Ngood > Nbest:
                best_ind = good_ind
                Nbest = Ngood
    
    if Nbest > 0:
        # Compile stem from circles
        keep = circles[best_ind,:]
        all_heights = sorted(list(set(list(keep[:,3]))))
        stem_data = []
        for h in all_heights:
            current_circs = keep[np.nonzero(keep[:,3]==h)[0],:]
            best_i = np.argmax(current_circs[:,4])
            stem_data.append(list(current_circs[best_i,:]))
        return stem_data
    else:
        return None
    
    # return found stem
    #if Nfound > 0:
    #    return params[0:Nfound,:]
    #else:
    #    return None

################################
# MergetoTrees: takes extracted stems (collections of stacked circles) and
# merges stems based on proximity etc.

def MergetoTrees(stem_data):
    stem_activeflag = [True]*len(stem_data)
    for i in range(len(stem_data)):
        if stem_activeflag[i] == False:
            continue
        for j in range(i+1,len(stem_data)):
            common_inds = [v for v in [d[5] for d in stem_data[i]] if v in [d[5] for d in stem_data[j]]]
            if len(common_inds) > 0:
                stem_data[i].extend(stem_data[j])
                stem_activeflag[j] = False
                all_heights = sorted(list(set([v[3] for v in stem_data[i]])))
                stem_data[i] = np.array(stem_data[i])
                stem_data_new = []
                for h in all_heights:
                    current_circs = stem_data[i][np.nonzero(stem_data[i][:,3]==h)[0],:]
                    best_i = np.argmax(current_circs[:,4])
                    stem_data_new.append(list(current_circs[best_i,:]))
                stem_data[i] = stem_data_new

    stem_data_final = []
    for i in range(len(stem_data)):
        if stem_activeflag[i] == False:
            continue
        stem_data_final.append(stem_data[i])
        
    return stem_data_final


############################################
# Function for extracting tree/stems points

def ExtractPointsusingStem(tree,X,Y,Z, dh=0.5,r_thresh_out = 0.05,r_thresh_in = 0.05,tree_size=3.0):
    
    xmin = min([tree[i][0] for i in range(len(tree))])-tree_size/2
    xmax = max([tree[i][0] for i in range(len(tree))])+tree_size/2
    ymin = min([tree[i][1] for i in range(len(tree))])-tree_size/2
    ymax = max([tree[i][1] for i in range(len(tree))])+tree_size/2
    zmin = min([tree[i][3] for i in range(len(tree))])
    zmax = max([tree[i][3] for i in range(len(tree))])
    xv = np.logical_and(X >= xmin, X < xmax)
    yv = np.logical_and(np.logical_and(Y >= ymin, Y < ymax),xv)
    inds = np.nonzero(yv)
    Xs = X[inds]
    Ys = Y[inds]
    Zs = Z[inds]
    tree_points = [Xs,Ys,Zs]
    stem_points = []
    for section in tree:
        indsz = np.nonzero(np.logical_and(Zs >= section[3]-dh/2, Zs < section[3]+dh/2))
        Xs2 = Xs[indsz]
        Ys2 = Ys[indsz]
        Zs2 = Zs[indsz]
        rel_points = np.stack((Xs2,Ys2),axis=1)-np.array([section[0],section[1]])
        r_all = np.linalg.norm(rel_points,axis=1)
        r_diff = r_all - section[2]
        good_ind = np.nonzero(np.logical_and(r_diff < r_thresh_out, r_diff > -r_thresh_in))[0]
        stem_points.append([Xs2[good_ind],Ys2[good_ind],Zs2[good_ind]])
    
    return (stem_points, tree_points)
    


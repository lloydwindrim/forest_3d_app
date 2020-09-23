'''
    Authors: Dr. Lloyd Windrim and Dr. Mitch Bryson
    Required packages: numpy, scipy, scikit-learn

    The primary purpose of this module is to remove and load the ground. The function removeGround() uses the TreePointCloud()
    class to remove the ground from the input pointcloud, and save the ground as a mesh (plyfile). The function
    load_ground_surface() can then be used to load the created ground mesh into python as points.

'''

import numpy as np
import os
from scipy.spatial import Delaunay
from scipy.spatial import KDTree
from plyfile import PlyData, PlyElement

# TreePointCloud - class to store pointcloud and associated surfaces
class TreePointCloud(object):

    def __init__(self, vertex, offset):

        self.offset = offset
        self.vertex = vertex

    # create_ground_points - estimate ground points based on grid minimas and median filtering
    def create_ground_points(self, grid_size=2.0):

        # create grid-based data structure for points
        xmin = np.min(self.vertex[:, 0])
        xmax = np.max(self.vertex[:, 0])
        ymin = np.min(self.vertex[:, 1])
        ymax = np.max(self.vertex[:, 1])
        nx = int((xmax - xmin) / grid_size + 1)
        ny = int((ymax - ymin) / grid_size + 1)

        grid_points = []
        for ix in range(nx):
            grid_points.append([])
            for iy in range(ny):
                grid_points[-1].append([])
        grid_mins = np.zeros((nx, ny))

        # cycle through and assign points into grids, grab minimas
        for v in self.vertex:  # put points in grid
            ix = int((v[0] - xmin) / grid_size)
            iy = int((v[1] - ymin) / grid_size)
            if ix >= 0 and ix < nx and iy >= 0 and iy < ny:
                grid_points[ix][iy].append(v)
        for ix in range(nx):  # get minima per grid
            for iy in range(ny):
                grid_points[ix][iy] = np.array(grid_points[ix][iy])
                if grid_points[ix][iy].shape[0] > 0:
                    grid_mins[ix, iy] = np.min(grid_points[ix][iy][:, 2])

        # add minimas to ground if they meet median-based outlier criteria
        self.ground_points = []
        for ix in range(nx):
            for iy in range(ny):
                vals = []
                for i in [-3, -2, -1, 0, 1, 2, 3]:
                    for j in [-3, -2, -1, 0, 1, 2, 3]:
                        i2 = i + ix
                        j2 = j + iy
                        if i2 >= 0 and i2 < nx and j2 >= 0 and j2 < ny:
                            if grid_points[i2][j2].shape[0] > 0:
                                vals.append(grid_mins[i2, j2])
                if len(vals) > 0:
                    medval = np.median(vals)
                    if grid_points[ix][iy].shape[0] > 0:
                        if abs(grid_mins[ix, iy] - medval) < 1.0:
                            ind = np.argmin(grid_points[ix][iy][:, 2])
                            self.ground_points.append(grid_points[ix][iy][ind, :])
                        else:
                            self.ground_points.append(
                                np.array([xmin + (ix + 0.5) * grid_size, ymin + (iy + 0.5) * grid_size, medval]))
        self.ground_points = np.array(self.ground_points)

    # save_ground_surface - output estimated ground surface as a triangular mesh
    def save_ground_surface(self, outputpath):

        # Build Delaunay triangulation
        tri = Delaunay(self.ground_points[:, 0:2])

        # Save output surface to ply file
        f = open(outputpath, "w")
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('comment %.3f\n' % (self.offset[0]))
        f.write('comment %.3f\n' % (self.offset[1]))
        f.write('comment %.3f\n' % (self.offset[2]))
        f.write('element vertex %d\n' % (self.ground_points.shape[0]))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('element face %d\n' % (tri.simplices.shape[0]))
        f.write('property list uchar int vertex_indices\n')
        f.write('end_header\n')
        for i in range(self.ground_points.shape[0]):
            f.write('%.4f %.4f %.4f\n' % (self.ground_points[i, 0], self.ground_points[i, 1], self.ground_points[i, 2]))
        for i in range(tri.simplices.shape[0]):
            f.write('3 %d %d %d\n' % (tri.simplices[i, 0], tri.simplices[i, 1], tri.simplices[i, 2]))
        f.close()


    # initialise_grid - initialise parameters and offset of a grid to store pointcloud for
    # further spatial processing
    def initialise_grid(self, grid_size=4.0):
        self.grid_size = grid_size
        self.xmin = np.min(self.vertex[:, 0])
        self.xmax = np.max(self.vertex[:, 0])
        self.ymin = np.min(self.vertex[:, 1])
        self.ymax = np.max(self.vertex[:, 1])
        self.nx = int((self.xmax - self.xmin) / grid_size + 1)
        self.ny = int((self.ymax - self.ymin) / grid_size + 1)

    # create_ground_grid - Grids ground points using nearest neighbours
    def create_ground_grid(self):
        self.ground_grid = np.zeros((self.nx, self.ny))
        kdtree = KDTree(self.ground_points[:, 0:2])
        for ix in range(self.nx):
            for iy in range(self.ny):
                (d, ind) = kdtree.query(
                    [self.xmin + (ix + 0.5) * self.grid_size, self.ymin + (iy + 0.5) * self.grid_size], k=4)
                self.ground_grid[ix, iy] = sum(self.ground_points[ind, 2] * d / sum(d))


def removeGround(vertex, offset=[0,0,0], returns=None, proc_path=None, name='', thresh=2.0, grid_size=4.0):

    # perform some tasks
    tpc = TreePointCloud(vertex, offset)

    tpc.create_ground_points(grid_size=2.0)
    if proc_path is not None:
        tpc.save_ground_surface(os.path.join(proc_path, name + '_ground_surface.ply'))

    # Build Gridded Data
    tpc.initialise_grid(grid_size=grid_size)
    tpc.create_ground_grid()

    # remove  ground surface
    tpc.vertex_gr = []
    xmin = np.min(tpc.vertex[:, 0])
    xmax = np.max(tpc.vertex[:, 0])
    ymin = np.min(tpc.vertex[:, 1])
    ymax = np.max(tpc.vertex[:, 1])
    nx = int((xmax - xmin) / grid_size + 1)
    ny = int((ymax - ymin) / grid_size + 1)

    returns_gr = []
    for idx, v in enumerate(tpc.vertex):  # put points in grid
        ix = int((v[0] - xmin) / grid_size)
        iy = int((v[1] - ymin) / grid_size)
        if ix >= 0 and ix < nx and iy >= 0 and iy < ny:
            if v[2] > (tpc.ground_grid[ix, iy] + thresh):
                tpc.vertex_gr.append(v)
                if returns is not None:
                    returns_gr.append(returns[idx])
    tpc.vertex_gr = np.array(tpc.vertex_gr)
    returns_gr = np.array(returns_gr)

    if len(np.shape(returns_gr))==1:
        returns_gr=returns_gr[:,np.newaxis]

    if returns is None:
        return tpc.vertex_gr
    else:
        return tpc.vertex_gr, returns_gr


# load_ground_surface - Load a ply file containing an estimated ground surface
def load_ground_surface(surface_path,offset=[0,0,0]):
    ground_points = []

    plydata_ground = PlyData.read(surface_path)
    offset_ground = np.array(
        [float(plydata_ground.comments[0]), float(plydata_ground.comments[1]), float(plydata_ground.comments[2])])
    offset_correction = offset_ground - offset
    for i in range(plydata_ground.elements[0].data.shape[0]):
        p = list(plydata_ground.elements[0].data[i])
        ground_points.append([p[j] + offset_correction[j] for j in [0, 1, 2]])
    ground_points = np.array(ground_points)

    return ground_points
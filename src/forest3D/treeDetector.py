# custom libraries
from forest3D import detection_tools,processLidar
from forest3D.object_detectors import detectObjects_yolov3 as detectObjects

# standard libaries
import numpy as np
import sys
import matplotlib.pyplot as plt
from skimage import exposure
import os

class RasterDetector():

    def __init__(self, raster_layers, support_window=[1,1,1],normalisation='rescale',doHisteq=[True,True,True],res=0.2,gridSize=[600, 600, 1000] ):

        self.raster_layers = ['blank','blank','blank']

        for i,layer_name in enumerate(raster_layers):
            self.raster_layers[i] = layer_name

        self.support_window = [1,1,1]

        for i,val in enumerate(support_window):
            self.support_window[i] = val

        self.normalisation = normalisation

        self.doHisteq = [1,1,1]
        for i, val in enumerate(doHisteq):
            self.doHisteq[i] = val

        self.res = res
        self.gridSize = np.array(gridSize)

    def sliding_window(self,detector_addr,xyz_data,colour_data=None,ground_pts=None,windowSize = [100,100],stepSize = 80,
                       classID=0,confidence_thresh=0.5,overlap_thresh=5,returnBoxes=False):

        if colour_data is None:
            xyz_clr_data = xyz_data
        elif len(np.shape(colour_data))==1:
            xyz_clr_data = np.hstack((xyz_data, colour_data[:,np.newaxis]))
        elif len(np.shape(colour_data)) == 2:
            xyz_clr_data = np.hstack((xyz_data, colour_data))

        box_store = np.zeros((1, 4))
        counter = 0

        for (x, y, window) in detection_tools.sliding_window_3d(xyz_clr_data, stepSize=stepSize,windowSize=windowSize):  # stepsize 100

            # track progress
            counter = counter + 1
            totalCount = len(range(int(np.min(xyz_data[:, 0])), int(np.max(xyz_data[:, 0])),stepSize)) * \
                         len(range(int(np.min(xyz_data[:, 1])), int(np.max(xyz_data[:, 1])),stepSize))
            sys.stdout.write("\r%d out of %d tiles" % (counter, totalCount))
            sys.stdout.flush()

            if window is not None:

                raster_stack,centre = self._rasterise(window, ground_pts=ground_pts)
                raster_stack = np.uint8(raster_stack*255)

                # use object detector to detect trees in raster
                [img, boxes, classes, scores] = detectObjects(raster_stack, addr_weights=os.path.join(detector_addr,'yolov3.weights'),
                                                              addr_confg=os.path.join(detector_addr,'yolov3.cfg'),MIN_CONFIDENCE=confidence_thresh)

                if np.shape(boxes)[0]:

                    # convert raster coordinates of bounding boxes to global x y coordinates
                    bb_coord = detection_tools.boundingBox_to_3dcoords(boxes_=boxes, gridSize_=self.gridSize[0:2], gridRes_=self.res,
                                                                      windowSize_=windowSize, pcdCenter_=centre)

                    # aggregate over windows
                    box_store = np.vstack((box_store, bb_coord[classes == classID, :]))

        box_store = box_store[1:, :]

        # remove overlapping boxes
        idx = detection_tools.find_unique_boxes2(box_store, overlap_thresh)
        box_store = box_store[idx, :]

        if returnBoxes:

            return box_store
        else:
            # # label points in pcd according to which bounding box they are in
            labels = detection_tools.label_pcd_from_bbox(xyz_clr_data, box_store[:, [1, 3, 0, 2]])

            return labels


    def rasterise(self,xyz_data,colour_data=None,ground_pts=None):

        if colour_data is None:
            xyz_clr_data = xyz_data
        elif len(np.shape(colour_data))==1:
            xyz_clr_data = np.hstack((xyz_data, colour_data[:,np.newaxis]))
        elif len(np.shape(colour_data)) == 2:
            xyz_clr_data = np.hstack((xyz_data, colour_data))

        raster_stack, centre = self._rasterise(xyz_clr_data, ground_pts=ground_pts)

        return raster_stack


    def _rasterise(self,data,ground_pts=None):

        # create raster layers
        raster1, centre = get_raster(self.raster_layers[0], data.copy(), self.support_window[0], self.res,
                                     self.gridSize, ground_pts=ground_pts)
        raster2, _ = get_raster(self.raster_layers[1], data.copy(), self.support_window[1], self.res,
                                self.gridSize, ground_pts=ground_pts)
        raster3, _ = get_raster(self.raster_layers[2], data.copy(), self.support_window[2], self.res,
                                self.gridSize, ground_pts=ground_pts)

        # normalise
        if self.normalisation == 'rescale':

            rasters_eq = []
            for i, raster in enumerate([raster1, raster2, raster3]):
                if self.raster_layers[i] is not 'blank':

                    plow, phigh = np.percentile(raster, (0, 100))
                    raster = exposure.rescale_intensity(raster, in_range=(plow, phigh))
                    rasters_eq.append(raster)
                else:
                    rasters_eq.append(raster)
            raster_stack = np.stack((rasters_eq[0], rasters_eq[1], rasters_eq[2]), axis=2)


        if self.normalisation == 'rescale+histeq':

            rasters_eq = []
            for i, raster in enumerate([raster1, raster2, raster3]):
                if self.raster_layers[i] is not 'blank':

                    if self.doHisteq[i]:
                        raster_eq = exposure.equalize_hist(raster)
                    else:
                        raster_eq = raster
                    plow, phigh = np.percentile(raster_eq, (0, 100))
                    raster_eq = exposure.rescale_intensity(raster_eq, in_range=(plow, phigh))
                    rasters_eq.append(raster_eq)
                else:
                    rasters_eq.append(raster)
            raster_stack = np.stack((rasters_eq[0], rasters_eq[1], rasters_eq[2]), axis=2)


        elif self.normalisation == 'cmap_jet':

            assert ((self.raster_layers[1] == 'blank') & (self.raster_layers[2] == 'blank'))

            cmap = plt.cm.jet
            norm = plt.Normalize(vmin=raster1.min(), vmax=raster1.max())
            raster_stack = cmap(norm(raster1))[..., 0:3]

        return raster_stack,centre


def get_raster(method_name,data,support_window,res,gridSize,ground_pts=None):

    if method_name == 'vertical_density':
        PCD = processLidar.ProcessPC(data[:, :3])
        PCD.occupancyGrid_Binary(res_=res, gridSize_=gridSize)
        PCD.vertical_density(support_window=support_window)

        return PCD.bev_verticalDensity,PCD._ProcessPC__centre[0:2]

    elif method_name == 'mean_colour1':
        PCD = processLidar.ProcessPC(data[:, :3],pc_returns=(data[:, 3]))
        PCD.occupancyGrid_Binary(res_=res, gridSize_=gridSize)
        PCD.occupancyGrid_Returns()
        PCD.mean_returns(support_window=support_window)

        return PCD.bev_meanReturn,PCD._ProcessPC__centre[0:2]

    elif method_name == 'mean_colour2':
        PCD = processLidar.ProcessPC(data[:, :3],pc_returns=(data[:, 4]))
        PCD.occupancyGrid_Binary(res_=res, gridSize_=gridSize)
        PCD.occupancyGrid_Returns()
        PCD.mean_returns(support_window=support_window)

        return PCD.bev_meanReturn,PCD._ProcessPC__centre[0:2]

    elif method_name == 'mean_colour3':
        PCD = processLidar.ProcessPC(data[:, :3],pc_returns=(data[:, 5]))
        PCD.occupancyGrid_Binary(res_=res, gridSize_=gridSize)
        PCD.occupancyGrid_Returns()
        PCD.mean_returns(support_window=support_window)

        return PCD.bev_meanReturn,PCD._ProcessPC__centre[0:2]

    elif method_name == 'max_colour1':
        PCD = processLidar.ProcessPC(data[:, :3],pc_returns=(data[:, 3]))
        PCD.occupancyGrid_Binary(res_=res, gridSize_=gridSize)
        PCD.occupancyGrid_Returns()
        PCD.max_returns(support_window=support_window)

        return PCD.bev_maxReturn,PCD._ProcessPC__centre[0:2]

    elif method_name == 'max_colour2':
        PCD = processLidar.ProcessPC(data[:, :3], pc_returns=(data[:, 4]))
        PCD.occupancyGrid_Binary(res_=res, gridSize_=gridSize)
        PCD.occupancyGrid_Returns()
        PCD.max_returns(support_window=support_window)

        return PCD.bev_maxReturn,PCD._ProcessPC__centre[0:2]

    elif method_name == 'max_colour3':
        PCD = processLidar.ProcessPC(data[:, :3], pc_returns=(data[:, 5]))
        PCD.occupancyGrid_Binary(res_=res, gridSize_=gridSize)
        PCD.occupancyGrid_Returns()
        PCD.max_returns(support_window=support_window)

        return PCD.bev_maxReturn,PCD._ProcessPC__centre[0:2]

    elif method_name == 'max_height':
        PCD = processLidar.ProcessPC(data[:, :3])
        PCD.occupancyGrid_Binary(res_=res, gridSize_=gridSize)
        PCD.max_height(support_window=support_window)

        return PCD.bev_maxHeight,PCD._ProcessPC__centre[0:2]

    elif method_name == 'canopy_height':
        PCD = processLidar.ProcessPC(data[:, :3])
        PCD.ground_normalise(ground_pts)
        PCD.occupancyGrid_Binary(res_=res, gridSize_=gridSize)
        PCD.max_height(support_window=support_window)

        return PCD.bev_maxHeight,PCD._ProcessPC__centre[0:2]

    elif method_name == 'blank':

        return 0.5 * np.ones((gridSize[:2])),None


'''
    Author: Dr. Lloyd Windrim
    Required packages: scikit-learn, scipy, numpy

    This library contains a class for individual pointclouds ( ProcessPC() ), and classes for lists of pointclouds
    ( database_points_pointlabels() , database_points() , database_binaryVoxels() , database_binaryVoxels_pointlabels() ).

    ProcessPC() is useful for various applications where you want a single pointcloud to be an object, and you wish you
    to mutate that object in different ways (e.g. voxelise, rasterise, normalise, rotate, etc.)

    The database classes are designed for machine learning applications. They are useful as iterators when you want to
    use several pointclouds to train (or predict with) a machine learning model (e.g. return the next batch of
    pointclouds to train on).

'''


import numpy as np
import copy
import random
from scipy import signal #only for bev vertical density
from scipy import ndimage #only for bev max height
from sklearn.neighbors import KDTree # only for ground normalisation

# Insert another class which, given a list of pointclouds, splits data into training and val upon init.
# Contains get batch, when called it, pulls random batch of pointclouds as list, loops through list using ProcessPC

def gaussian_kernel(size,mu=0.0,sigma=1.0):
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    d = np.sqrt(x * x + y * y)
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    return g

class database_points_pointlabels():

    def __init__(self, xyz_list=None,labels=None,returns=None, numClasses=None, batchsize=None ):

        self.data_num = len(xyz_list)
        if numClasses is None:
            self.nb_class = np.size(np.unique(labels[0])) # only counts number of classes in first pcd
        else:
            self.nb_class = numClasses


        self.pc_list = []
        for i in range(len(xyz_list)):
            if (labels is None) & (returns is None):
                self.pc_list.append(ProcessPC(xyz_data=xyz_list[i]))
            elif (labels is None) & (returns is not None):
                self.pc_list.append(ProcessPC(xyz_data=xyz_list[i], pc_returns=returns[i]))
            elif (labels is not None) & (returns is None):
                self.pc_list.append(ProcessPC(xyz_data=xyz_list[i], pc_labels=labels[i]))
            else:
                self.pc_list.append(ProcessPC(xyz_data=xyz_list[i],pc_labels=labels[i],pc_returns=returns[i]))

        if labels is None:
            self.flag_label = 0
        else:
            self.flag_label = 1
        if returns is None:
            self.flag_return = 0
        else:
            self.flag_return = 1


        if batchsize is not None:
            self.batchsize = batchsize
        else:
            self.batchsize = self.data_num
        self.current_batch = np.arange(self.batchsize)


    def next_batch( self, augment=False, pre_process=False, addAxis=False ):
        # augments the current batch once
        # use with network_pointnet_bs1 (only works for batchsize of 1)


        pc_temp = copy.deepcopy(self.pc_list[self.current_batch[0]])

        if augment==True:
            pc_temp.augmentation_Rotation()
            pc_temp.augmentation_Flipping()

        pc_batch = pc_temp.pc.copy()
        if addAxis: # if using bsK, want output to be K x 1 x N x 3, where k=1
            pc_batch = pc_batch[np.newaxis,...]
            pc_batch = pc_batch[np.newaxis, ...]

        if pre_process:
            pc_batch[0, ...] -= np.min(pc_batch[i, ...], axis=0)
            pc_batch[0, ...] /= np.max(pc_batch[i, ...])

        if self.flag_label == 1:
            labels = np.array(pc_temp.pc_labels)

            labels_onehot = np.zeros((len(labels), self.nb_class))
            labels_onehot[np.arange(len(labels)), labels] = 1

        if self.flag_return == 1:
            pc_returns_batch = np.array(pc_temp.pc_returns)


        # update current batch
        self.current_batch += self.batchsize
        self.current_batch[self.current_batch >= self.data_num] = \
            self.current_batch[self.current_batch >= self.data_num] - self.data_num



        if (self.flag_label == 1) & (self.flag_return == 0):
            return pc_batch, labels_onehot, labels
        elif (self.flag_label == 0) & (self.flag_return == 1):
            return pc_batch, pc_returns_batch
        elif (self.flag_label == 1) & (self.flag_return == 1):
            return pc_batch, labels_onehot, labels, pc_returns_batch
        else:
            return pc_batch

    def next_batch2( self, augment=False, numAugs=0, pre_process=False, angle_x_randLim=0, angle_y_randLim=0, normalise=False, newAxis_loc=1 ):
        # appends numAugs different augmentations to the current batch

        n_points = self.pc_list[self.current_batch[0]].pc.shape[0]
        pc_batch = np.empty( ( [self.batchsize*(numAugs+1),n_points,3] ) )
        labels = np.empty( ( [self.batchsize*(numAugs+1),n_points] ) )
        returns_batch = np.empty( ( [self.batchsize*(numAugs+1),n_points] ) )
        for j in range(numAugs+1):
            for i in range(self.batchsize):
                pc_temp = copy.deepcopy(self.pc_list[self.current_batch[i]])

                if (augment==True)&(j>0): # leave one set un-augmented
                    pc_temp.augmentation_Rotation(angle_x_randLim=angle_x_randLim, angle_y_randLim=angle_y_randLim)
                    pc_temp.augmentation_Flipping()
                if normalise:
                    pc_temp.normalisation()
                pc_batch[(j*self.batchsize)+i,...] = pc_temp.pc.copy()
                labels[(j*self.batchsize)+i,...] = pc_temp.pc_labels.copy()
                if self.flag_return == 1:
                    returns_batch[(j * self.batchsize) + i, ...] = pc_temp.pc_returns.copy()


        # pre-process
        if pre_process:
            pc_batch[0, ...] -= np.min(pc_batch[i, ...], axis=0)
            pc_batch[0, ...] /= np.max(pc_batch[i, ...])

        if newAxis_loc == 1:
            pc_batch = pc_batch[:, np.newaxis, ...]
        elif newAxis_loc == 0:
            pc_batch = pc_batch[np.newaxis, ...]


        #labels = np.array(pc_temp.pc_labels)
        #labels = np.tile(labels[:,np.newaxis],(1,numAugs+1))

        labels_onehot = np.zeros((self.batchsize*(numAugs+1) , n_points , self.nb_class))
        #labels_onehot[:,np.arange(n_points), labels.astype(np.int).T] = 1
        xv, yv = np.meshgrid(np.arange(0, (self.batchsize*(numAugs+1))), np.arange(0, n_points))
        labels_onehot[np.ravel(xv), np.ravel(yv), np.ravel( labels.astype(np.int).T )] = 1


        #labels = np.tile(labels[np.newaxis,:],[numAugs+1,1])

        # update current batch
        self.current_batch += self.batchsize
        self.current_batch[self.current_batch >= self.data_num] = \
            self.current_batch[self.current_batch >= self.data_num] - self.data_num



        if self.flag_return == 1:
            return pc_batch, labels_onehot,labels, returns_batch
        else:
            return pc_batch, labels_onehot,labels


    def get_pc(self, idx=[1], augment=False, angle_x=0, angle_y=0, angle_z=30, pre_process=False, addAxis=False, normalise=False, newAxis_loc=1 ):
        # default not to augment, but if so you can specify the rotations. Default rotation only about z

        n_points = self.pc_list[idx[0]].pc.shape[0]
        pc_batch = np.empty(([len(idx), n_points, 3]))
        if self.flag_label==1:
            labels = np.empty(([len(idx), n_points]))
        if self.flag_return == 1:
            returns_batch = np.empty(([len(idx), n_points]))
        for i in range(len(idx)):
            pc_temp = copy.deepcopy(self.pc_list[idx[i]])

            if augment==True:
                pc_temp.augmentation_Rotation(angle_x=angle_x, angle_y=angle_y, angle_z=angle_z)
                #pc_temp.augmentation_Flipping()
            if normalise:
                pc_temp.normalisation()
            pc_batch[i,...] = pc_temp.pc.copy()
            if self.flag_label == 1:
                labels[i,:] = np.array(pc_temp.pc_labels)
            if self.flag_return == 1:
                returns_batch[i,:] = np.array(pc_temp.pc_returns)

        if addAxis:
            if newAxis_loc == 1:
                pc_batch = pc_batch[:, np.newaxis, ...]
            elif newAxis_loc == 0:
                pc_batch = pc_batch[np.newaxis, ...]

        if self.flag_label == 1:
            labels_onehot = np.zeros((len(idx), n_points, self.nb_class))

            xv, yv = np.meshgrid(np.arange(0, (len(idx))), np.arange(0, n_points))
            labels_onehot[np.ravel(xv), np.ravel(yv), np.ravel(labels.astype(np.int).T)] = 1

        if (self.flag_label == 1) & (self.flag_return == 0):
            return pc_batch, labels_onehot, labels
        elif (self.flag_label == 0) & (self.flag_return == 1):
            return pc_batch, returns_batch
        elif (self.flag_label == 1) & (self.flag_return == 1):
            return pc_batch, labels_onehot, labels, returns_batch
        else:
            return pc_batch

    def indexed_batch( self, idx=[1], augment=False, numAugs=0, pre_process=False, angle_x_randLim=0, angle_y_randLim=0, normalise=False, newAxis_loc=1, adapt_num_classes=False ):
        # appends numAugs different augmentations to the current batch



        n_points = self.pc_list[idx[0]].pc.shape[0]
        pc_batch = np.empty( ( [len(idx)*(numAugs+1),n_points,3] ) )
        labels = np.empty( ( [len(idx)*(numAugs+1),n_points] ) )
        returns_batch = np.empty(([len(idx)*(numAugs+1),n_points]))
        for j in range(numAugs+1):
            for i in range(len(idx)):
                pc_temp = copy.deepcopy(self.pc_list[idx[i]])

                if (augment==True)&(j>0): # leave one set un-augmented
                    pc_temp.augmentation_Rotation(angle_x_randLim=angle_x_randLim, angle_y_randLim=angle_y_randLim)
                    pc_temp.augmentation_Flipping()
                    #pc_temp.augmentation_Shuffle()
                if normalise:
                    pc_temp.normalisation()
                pc_batch[(j*len(idx))+i,...] = pc_temp.pc.copy()
                labels[(j*len(idx))+i,...] = pc_temp.pc_labels.copy()
                if self.flag_return == 1:
                    returns_batch[(j*len(idx))+i,...] = pc_temp.pc_returns.copy()


        # pre-process
        if pre_process:
            pc_batch[0, ...] -= np.min(pc_batch[i, ...], axis=0)
            pc_batch[0, ...] /= np.max(pc_batch[i, ...])

        if newAxis_loc == 1:
            pc_batch = pc_batch[:,np.newaxis, ...]
        elif newAxis_loc == 0:
            pc_batch = pc_batch[np.newaxis, ...]


        #labels = np.array(pc_temp.pc_labels)
        #labels = np.tile(labels[:,np.newaxis],(1,numAugs+1))

        if adapt_num_classes:  # allows number of classes (and hence size of onehot) to be modified each batch
            self.nb_class = len(np.unique(labels))

        labels_onehot = np.zeros((len(idx)*(numAugs+1) , n_points , self.nb_class))
        #labels_onehot[:,np.arange(n_points), labels.astype(np.int).T] = 1
        xv, yv = np.meshgrid(np.arange(0, (len(idx)*(numAugs+1))), np.arange(0, n_points))
        labels_onehot[np.ravel(xv), np.ravel(yv), np.ravel( labels.astype(np.int).T )] = 1


        #labels = np.tile(labels[np.newaxis,:],[numAugs+1,1])

        if self.flag_return == 1:
            return pc_batch, labels_onehot,labels, returns_batch
        else:
            return pc_batch, labels_onehot,labels


    def reset_batch(self):
        """ Resets the current batch to the beginning.

        """

        self.current_batch = np.arange(self.batchsize)


    def shuffle(self):
        """ Randomly permutes all dataSamples (and corresponding targets).

        """

        random.shuffle(self.pc_list)

class database_points():

    def __init__(self, xyz_list=None,labels=None, batchsize=None ):

        self.data_num = len(xyz_list)
        self.nb_class = np.max(labels)+1
        self.first_batch = 1
        self.current_batch = []

        self.labels = labels
        self.pc_list = []
        for i in range(len(xyz_list)):
            self.pc_list.append(ProcessPC(xyz_list[i]))

        if batchsize is not None:
            self.batchsize = batchsize
        else:
            self.batchsize = self.data_num
        self.current_batch = np.arange(self.batchsize)


    def next_batch( self, augment=False, pre_process=False ):
        # augments the current batch once

        if self.first_batch:
            self.current_batch = (np.arange(self.batchsize)).tolist()
            self.first_batch = 0
        else:
            self.current_batch = (np.array(self.current_batch) + self.batchsize).tolist()
            if self.current_batch[-1] > (self.data_num - self.batchsize):
                self.first_batch = 1


        pc_temp = copy.deepcopy(self.pc_list[self.current_batch[0]])

        if augment==True:
            pc_temp.augmentation_Rotation()
            pc_temp.augmentation_Flipping()

        pc_batch = pc_temp.pc.copy()
        pc_batch = pc_batch[np.newaxis,...]

        # pre-process - scale between [-1,1]
        #if pre_process:
            #og_batch = 2*(og_batch-0.5)

        labels = np.array(self.labels)
        labels = labels[self.current_batch]

        labels_onehot = np.zeros((len(labels), self.nb_class))
        labels_onehot[np.arange(len(labels)), labels] = 1

        # update current batch
        self.current_batch += self.batchsize
        self.current_batch[self.current_batch >= self.data_num] = \
            self.current_batch[self.current_batch >= self.data_num] - self.data_num


        return pc_batch, labels_onehot, labels

    def next_batch2( self, augment=False, numAugs=0, pre_process=False ):
        # appends numAugs different augmentations to the current batch

        if self.first_batch:
            self.current_batch = (np.arange(self.batchsize)).tolist()
            self.first_batch = 0
        else:
            self.current_batch = (np.array(self.current_batch) + self.batchsize).tolist()
            if self.current_batch[-1] > (self.data_num - self.batchsize):
                self.first_batch = 1

        n_points = self.pc_list[self.current_batch[0]].pc.shape[0]
        pc_batch = np.empty( ( [self.batchsize*(numAugs+1),n_points,3] ) )
        for j in range(numAugs+1):
            pc_temp = copy.deepcopy(self.pc_list[self.current_batch[0]])

            if (augment==True)&(j>1): # leave one set un-augmented
                pc_temp.augmentation_Rotation()
                pc_temp.augmentation_Flipping()
            pc_batch[j,...] = pc_temp.pc.copy()

        # pre-process - scale between [-1,1]
        #if pre_process:
            #og_batch = 2*(og_batch-0.5)

        labels = np.array(self.labels)
        labels = labels[self.current_batch]
        labels = np.tile(labels,(numAugs+1))

        labels_onehot = np.zeros((len(labels), self.nb_class))
        labels_onehot[np.arange(len(labels)), labels] = 1

        # update current batch
        self.current_batch += self.batchsize
        self.current_batch[self.current_batch >= self.data_num] = \
            self.current_batch[self.current_batch >= self.data_num] - self.data_num


        return pc_batch, labels_onehot, labels

    def get_pc(self, idx=[1], augment=False, angle_x=0, angle_y=0, angle_z=30):
        # default not to augment, but if so you can specify the rotations. Default rotation only about z


        pc_temp = copy.deepcopy(self.pc_list[idx[0]])

        if augment==True:
            pc_temp.augmentation_Rotation(angle_x=angle_x, angle_y=angle_y, angle_z=angle_z)
            #pc_temp.augmentation_Flipping()
        pc_batch = pc_temp.pc.copy()
        pc_batch = pc_batch[np.newaxis, ...]

        labels = np.array(self.labels)
        labels = labels[idx]

        labels_onehot = np.zeros((len(labels), self.nb_class))
        labels_onehot[np.arange(len(labels)), labels] = 1

        return pc_batch, labels_onehot, labels

    def reset_batch(self):
        """ Resets the current batch to the beginning.

        """

        self.current_batch = np.arange(self.batchsize)


    def shuffle(self):
        """ Randomly permutes all dataSamples (and corresponding targets).

        """

        if self.labels is not None:
            zipped = list(zip(self.pc_list, self.labels))
            random.shuffle(zipped)
            self.pc_list, self.labels = list(zip(*zipped))
        else:
            random.shuffle(self.pc_list)
    
class database_binaryVoxels_pointlabels():

    def __init__(self, xyz_list=None,labels=None,returns=None, res=0.1, gridSize=np.array( (32,32,32) ), numClasses=None, batchsize=None):

        # make sure to input more than one pcd
        self.data_num = len(xyz_list)
        if numClasses is None:
            self.nb_class = np.size(np.unique(labels[0])) # only counts number of classes in first pcd
        else:
            self.nb_class = numClasses


        self.pc_list = []
        for i in range(len(xyz_list)):
            if (labels is None) & (returns is None):
                self.pc_list.append(ProcessPC(xyz_data=xyz_list[i]))
            elif (labels is None) & (returns is not None):
                self.pc_list.append(ProcessPC(xyz_data=xyz_list[i], pc_returns=returns[i]))
            elif (labels is not None) & (returns is None):
                self.pc_list.append(ProcessPC(xyz_data=xyz_list[i], pc_labels=labels[i]))
            else:
                self.pc_list.append(ProcessPC(xyz_data=xyz_list[i],pc_labels=labels[i],pc_returns=returns[i]))
        self.res=res
        self.gridSize=gridSize
        if labels is None:
            self.flag_label = 0
        else:
            self.flag_label = 1
        if returns is None:
            self.flag_return = 0
        else:
            self.flag_return = 1


        if batchsize is not None:
            self.batchsize = batchsize
        else:
            self.batchsize = self.data_num
        self.current_batch = np.arange(self.batchsize)


    def next_batch( self, augment=False, pre_process=False, outputOffset=False ):
        # augments the current batch once



        og_batch = np.empty( ( [self.batchsize,1] + self.gridSize.tolist() ) )
        offset = np.empty(([self.batchsize] + [3]))
        if self.flag_label == 1:
            og_labels_batch = np.empty(([self.batchsize, self.nb_class+1] + self.gridSize.tolist())) #+1 for free space
        if self.flag_return == 1:
            og_returns_batch = np.empty(([self.batchsize, 1] + self.gridSize.tolist()))
        for i in range(self.batchsize):
            pc_temp = copy.deepcopy(self.pc_list[self.current_batch[i]])

            if augment==True:
                pc_temp.augmentation_Rotation( )
                pc_temp.augmentation_Flipping( )
            pc_temp.occupancyGrid_Binary( res_=self.res, gridSize_=self.gridSize )
            og_batch[i,...] = pc_temp.og.copy()
            offset[i, :] = pc_temp._ProcessPC__centre
            if self.flag_label == 1:
                pc_temp.occupancyGrid_Labels()
                og_labels_batch[i, ...] = pc_temp.og_labels.copy()
            if self.flag_return == 1:
                pc_temp.occupancyGrid_Returns()
                og_returns_batch[i,...] = pc_temp.og_returns.copy()

        # pre-process - scale between [-1,1]
        if pre_process:
            og_batch = 2*(og_batch-0.5)


        # update current batch
        self.current_batch += self.batchsize
        self.current_batch[self.current_batch >= self.data_num] = \
            self.current_batch[self.current_batch >= self.data_num] - self.data_num



        if outputOffset is False:
            if (self.flag_label == 1)&(self.flag_return == 0):
                return og_batch, og_labels_batch
            elif (self.flag_label == 0)&(self.flag_return == 1):
                return og_batch, og_returns_batch
            elif (self.flag_label == 1)&(self.flag_return == 1):
                return og_batch, og_labels_batch, og_returns_batch
            else:
                return og_batch
        else:
            if (self.flag_label == 1)&(self.flag_return == 0):
                return og_batch, og_labels_batch, offset
            elif (self.flag_label == 0)&(self.flag_return == 1):
                return og_batch, og_returns_batch, offset
            elif (self.flag_label == 1)&(self.flag_return == 1):
                return og_batch, og_labels_batch, og_returns_batch, offset
            else:
                return og_batch, offset

    def next_batch2( self, augment=False, numAugs=0, pre_process=False,angle_x_randLim=0, angle_y_randLim=0 ):
        # appends numAugs different augmentations to the current batch


        og_batch = np.empty( ( [self.batchsize*(numAugs+1),1] + self.gridSize.tolist() ) )
        og_labels_batch = np.empty( ( [self.batchsize*(numAugs+1),self.nb_class+1] + self.gridSize.tolist() ) )
        og_returns_batch = np.empty(([self.batchsize * (numAugs + 1), 1] + self.gridSize.tolist()))
        for j in range(numAugs+1):
            for i in range(self.batchsize):
                pc_temp = copy.deepcopy(self.pc_list[self.current_batch[i]])

                # augment pointcloud
                if (augment==True)&(j>0): # leave one set un-augmented
                    pc_temp.augmentation_Rotation(angle_x_randLim=angle_x_randLim, angle_y_randLim=angle_y_randLim )
                    pc_temp.augmentation_Flipping( )
                # occupancy grid
                pc_temp.occupancyGrid_Binary( res_=self.res, gridSize_=self.gridSize )
                og_batch[(j*self.batchsize)+i,...] = pc_temp.og.copy()
                # labelled occupancy grid
                pc_temp.occupancyGrid_Labels()
                og_labels_batch[(j*self.batchsize)+i, ...] = pc_temp.og_labels.copy()
                # occupancy grid with returns
                if self.flag_return == 1:
                    pc_temp.occupancyGrid_Returns()
                    og_returns_batch[(j*self.batchsize)+i,...] = pc_temp.og_returns.copy()

        # pre-process - scale between [-1,1]
        if pre_process:
            og_batch = 2*(og_batch-0.5)


        # update current batch
        self.current_batch += self.batchsize
        self.current_batch[self.current_batch >= self.data_num] = \
            self.current_batch[self.current_batch >= self.data_num] - self.data_num


        if self.flag_return == 1:
            return og_batch, og_labels_batch, og_returns_batch
        else:
            return og_batch, og_labels_batch

    def get_pc(self, idx=[1,2,3], augment=False, angle_x=0, angle_y=0, angle_z=30, angle_x_randLim=0, angle_y_randLim=0, outputOffset=False  ):
        # default not to augment, but if so you can specify the rotations. Default rotation only about z. set to none for random rotation
        # useful for inference because doesnt need labels

        og_batch = np.empty( ( [len(idx),1] + self.gridSize.tolist() ) )
        offset = np.empty( ( [len(idx)] + [3] ) )
        if self.flag_label == 1:
            og_labels_batch = np.empty(([len(idx), self.nb_class+1] + self.gridSize.tolist()))
        if self.flag_return == 1:
            og_returns_batch = np.empty(([len(idx), 1] + self.gridSize.tolist()))
        for i in range(len(idx)):
            pc_temp = copy.deepcopy(self.pc_list[idx[i]])

            if augment==True:
                pc_temp.augmentation_Rotation(angle_x=angle_x, angle_y=angle_y, angle_z=angle_z, angle_x_randLim=angle_x_randLim, angle_y_randLim=angle_y_randLim )
                #pc_temp.augmentation_Flipping()
            pc_temp.occupancyGrid_Binary( res_=self.res, gridSize_=self.gridSize )
            og_batch[i,...] = pc_temp.og.copy()
            offset[i,:] = pc_temp._ProcessPC__centre
            if self.flag_label == 1:
                pc_temp.occupancyGrid_Labels()
                og_labels_batch[i,...] = pc_temp.og_labels.copy()
            if self.flag_return == 1:
                pc_temp.occupancyGrid_Returns()
                og_returns_batch[i,...] = pc_temp.og_returns.copy()

        if outputOffset is False:
            if (self.flag_label == 1)&(self.flag_return == 0):
                return og_batch, og_labels_batch
            elif (self.flag_label == 0)&(self.flag_return == 1):
                return og_batch, og_returns_batch
            elif (self.flag_label == 1)&(self.flag_return == 1):
                return og_batch, og_labels_batch, og_returns_batch
            else:
                return og_batch
        else:
            if (self.flag_label == 1)&(self.flag_return == 0):
                return og_batch, og_labels_batch, offset
            elif (self.flag_label == 0)&(self.flag_return == 1):
                return og_batch, og_returns_batch, offset
            elif (self.flag_label == 1)&(self.flag_return == 1):
                return og_batch, og_labels_batch, og_returns_batch, offset
            else:
                return og_batch, offset


    def reset_batch(self):
        """ Resets the current batch to the beginning.

        """

        self.current_batch = np.arange(self.batchsize)


    def shuffle(self):
        """ Randomly permutes all dataSamples (and corresponding targets).

        """

        random.shuffle(self.pc_list)


class database_binaryVoxels(): # onehot, object labels

    def __init__(self, xyz_list=None,labels=None, res=0.1, gridSize=np.array( (32,32,32) ), batchsize=None):

        self.data_num = len(xyz_list)
        self.nb_class = np.max(labels)+1

        self.labels = labels
        self.pc_list = []
        for i in range(len(xyz_list)):
            self.pc_list.append(ProcessPC(xyz_list[i]))
        self.res=res
        self.gridSize=gridSize

        if batchsize is not None:
            self.batchsize = batchsize
        else:
            self.batchsize = self.data_num
        self.current_batch = np.arange(self.batchsize)



    def next_batch( self, augment=False, pre_process=False ):
        # augments the current batch once

        og_batch = np.empty( ( [self.batchsize,1] + self.gridSize.tolist() ) )
        for i in range(self.batchsize):
            pc_temp = copy.deepcopy(self.pc_list[self.current_batch[i]])

            if augment==True:
                pc_temp.augmentation_Rotation()
                pc_temp.augmentation_Flipping()
            pc_temp.occupancyGrid_Binary( res_=self.res, gridSize_=self.gridSize )
            og_batch[i,...] = pc_temp.og.copy()

        # pre-process - scale between [-1,1]
        if pre_process:
            og_batch = 2*(og_batch-0.5)

        labels = np.array(self.labels)
        labels = labels[self.current_batch]

        labels_onehot = np.zeros((len(labels), self.nb_class))
        labels_onehot[np.arange(len(labels)), labels] = 1


        # update current batch
        self.current_batch += self.batchsize
        self.current_batch[self.current_batch >= self.data_num] = \
            self.current_batch[self.current_batch >= self.data_num] - self.data_num


        return og_batch, labels_onehot, labels

    def next_batch2( self, augment=False, numAugs=0, pre_process=False ):
        # appends numAugs different augmentations to the current batch



        og_batch = np.empty( ( [self.batchsize*(numAugs+1),1] + self.gridSize.tolist() ) )
        for j in range(numAugs+1):
            for i in range(self.batchsize):
                pc_temp = copy.deepcopy(self.pc_list[self.current_batch[i]])

                if (augment==True)&(j>1): # leave one set un-augmented
                    pc_temp.augmentation_Rotation()
                    pc_temp.augmentation_Flipping()
                pc_temp.occupancyGrid_Binary( res_=self.res, gridSize_=self.gridSize )
                og_batch[(j*self.batchsize)+i,...] = pc_temp.og.copy()

        # pre-process - scale between [-1,1]
        if pre_process:
            og_batch = 2*(og_batch-0.5)

        labels = np.array(self.labels)
        labels = labels[self.current_batch]
        labels = np.tile(labels,(numAugs+1))

        labels_onehot = np.zeros((len(labels), self.nb_class))
        labels_onehot[np.arange(len(labels)), labels] = 1

        # update current batch
        self.current_batch += self.batchsize
        self.current_batch[self.current_batch >= self.data_num] = \
            self.current_batch[self.current_batch >= self.data_num] - self.data_num

        return og_batch, labels_onehot, labels

    def get_pc(self, idx=[1,2,3], augment=False, angle_x=0, angle_y=0, angle_z=30):
        # default not to augment, but if so you can specify the rotations. Default rotation only about z

        og_batch = np.empty( ( [len(idx),1] + self.gridSize.tolist() ) )
        for i in range(len(idx)):
            pc_temp = copy.deepcopy(self.pc_list[idx[i]])

            if augment==True:
                pc_temp.augmentation_Rotation(angle_x=angle_x, angle_y=angle_y, angle_z=angle_z)
                #pc_temp.augmentation_Flipping()
            pc_temp.occupancyGrid_Binary( res_=self.res, gridSize_=self.gridSize )
            og_batch[i,...] = pc_temp.og.copy()

        labels = np.array(self.labels)
        labels = labels[idx]

        labels_onehot = np.zeros((len(labels), self.nb_class))
        labels_onehot[np.arange(len(labels)), labels] = 1

        return og_batch, labels_onehot, labels


    def reset_batch(self):
        """ Resets the current batch to the beginning.

        """

        self.current_batch = np.arange(self.batchsize)


    def shuffle(self):
        """ Randomly permutes all dataSamples (and corresponding targets).

        """

        if self.labels is not None:
            zipped = list(zip(self.pc_list, self.labels))
            random.shuffle(zipped)
            self.pc_list, self.labels = list(zip(*zipped))
        else:
            random.shuffle(self.pc_list)



class ProcessPC():
# this class is designed to store one pointcloud.
# it can augment that pointcloud and convert it to an occupancy grid

    def __init__( self, xyz_data = None , pc_labels=None , pc_returns=None ):
    # accepts N x 3 pointcloud array (x,y,z) and stores it

        self.og = [] # occupancy grid
        self.og_labels = []
        self.og_returns = []
        self.raster = []
        if np.shape(xyz_data)[1]==3:
            self.pc = xyz_data # pointcloud
        else:
            raise ValueError('Input pointcloud incorrect.')
        self.pc_labels = pc_labels
        self.pc_returns = pc_returns
        self.bev_verticalDensity = []
        self.bev_maxHeight = []
        self.bev_meanReturn = []
        self.__flag_recentred = False
        self.__index_x = 0
        self.__index_y = 0
        self.__index_z = 0
        self.__centre = [0,0,0]

    def recentre( self, centre_ ):

        self.pc -= centre_
        self.__flag_recentred = True
        self.__centre = centre_

    def occupancyGrid_Binary( self, res_=0.1, gridSize_=np.array( (32,32,32) ), centre=None ):
        """ Convert point cloud to occupancy grid representation
            Assumes point cloud has been recentred
            - input:
                   pc_: point cloud Mx3 numpy matrix
                   res_: resolution of each cell in metres
                   gridSize_: size of the occupancy grid (DEFAULT: 32x32x32)
            - output:
              ---: occupancy grid is a class variable
              Note: this fnc modifies the input pcd, so make a copy
        """

        # initialise grid
        self.og = np.zeros( gridSize_ )
        # recentre the point cloud about the mean if it is not done already
        if not self.__flag_recentred:
            if centre is None: # no centre specified
                c = np.mean( self.pc, axis=0 )
                c[2] = np.max( self.pc[:,2] )
                self.recentre( c  )
                if isinstance(res_,(list)):
                    h_offset = gridSize_[2] / 2. * res_[2]
                else:
                    h_offset = gridSize_[2] / 2. * res_
                self.pc[:, 2] += h_offset # additional offset
                self.__centre[2] -= h_offset # update offset info
            else:
                c = centre   # centre specified
                self.recentre(c)
        # get index of points within grid
        if isinstance(res_,(list)): # possible to specify different res for each dimension
            self.__index_x = np.array(np.clip(np.floor((self.pc[:, 0] - (-gridSize_[0] / 2. * res_[0])) / res_[0]), 0, gridSize_[0] - 1), dtype=int)
            self.__index_y = np.array(np.clip(np.floor((self.pc[:, 1] - (-gridSize_[1] / 2. * res_[1])) / res_[1]), 0, gridSize_[1] - 1), dtype=int)
            self.__index_z = np.array(np.clip(np.floor((self.pc[:, 2] - (-gridSize_[2] / 2. * res_[2])) / res_[2]), 0, gridSize_[2] - 1), dtype=int)
        else:
            self.__index_x = np.array( np.clip( np.floor( ( self.pc[:,0]-(-gridSize_[0]/2.*res_) )/res_ ), 0, gridSize_[0]-1 ), dtype=int)
            self.__index_y = np.array( np.clip( np.floor( ( self.pc[:,1]-(-gridSize_[1]/2.*res_) )/res_ ), 0, gridSize_[1]-1 ), dtype=int)
            self.__index_z = np.array( np.clip( np.floor( ( self.pc[:,2]-(-gridSize_[2]/2.*res_) )/res_ ), 0, gridSize_[2]-1 ), dtype=int)
        # set cells to occupied
        self.og[self.__index_x,self.__index_y,self.__index_z] = 1.

        #self.og_labels = np.zeros( np.hstack( (3,gridSize_) ) , dtype=np.int) #2 is for one-hot, two_classes
        self.og_labels = np.zeros(np.hstack((3, gridSize_)), dtype=np.int)
        self.og_returns = np.zeros(  gridSize_  )



    def og_label_util(self, cube, class_label, occupancy):
        # cube - which cube to work on
        # class_label - class number of point
        # occupancy - binary
        self.og_labels[cube, self._ProcessPC__index_x[self.pc_labels == class_label], self._ProcessPC__index_y[
            self.pc_labels == class_label],
                       self._ProcessPC__index_z[self.pc_labels == class_label]] = occupancy  # foliage


    def occupancyGrid_Labels( self ):

        # cube 0: background, 1: foliage_clutter, 2:stem
        self.og_labels[0, ...] = 1
        self.og_label_util(1, 0, 1)
        self.og_label_util(1, 3, 1)
        self.og_label_util(0, 0, 0)
        self.og_label_util(0, 3, 0)
        self.og_label_util(2, 1, 1)
        self.og_label_util(2, 2, 1)
        self.og_label_util(0, 1, 0)
        self.og_label_util(1, 1, 0)
        self.og_label_util(0, 2, 0)
        self.og_label_util(1, 2, 0)



    def occupancyGrid_Returns( self ):

        self.og_returns[self.__index_x, self.__index_y, self.__index_z] = self.pc_returns


    def writeOG(self, filename_='/tmp/og.txt'):
        f = open(filename_, 'w')
        for i in range(len(self.__index_x)):
            f.write(
                str(self.__index_x[i]) + ',' + str(self.__index_y[i]) + ',' + str(self.__index_z[i]) + ',' + str(1) + '\n')
        f.close()


    def augmentation_Rotation( self, angle_x=None, angle_y=None, angle_z=None, angle_x_randLim=10, angle_y_randLim=10  ):
        """ Augment the input point cloud by randomly rotating it
            - input:
                pc_: point cloud Mx3 numpy matrix
                rotation (degrees): default rotation is random. Put 0 if you want to fix rotation about an axis
                *add in z lim, but set default to full circle
            - output:
                ---: augmentations are applied in place
        """

        # randomly sample angles
        if angle_x==None:
            angle_x = ( np.random.rand(1)-0.5 )*np.deg2rad( float(angle_x_randLim) ) # default max roll, pitch rotation is 10.
        else:
            angle_x = np.deg2rad( float(angle_x) )

        if angle_y == None:
            angle_y = (np.random.rand(1) - 0.5) * np.deg2rad( float(angle_y_randLim) )
        else:
            angle_y = np.deg2rad( float(angle_y) )

        if angle_z == None:
            angle_z = np.random.rand(1)[0]*2.*np.pi # rotate full circle
        else:
            angle_z = np.deg2rad(float(angle_z))

        # generation rotation matrix
        Rx = self.__RotationX( angle_x )
        Ry = self.__RotationX( angle_y )
        Rz = self.__RotationZ( angle_z )
        R = np.dot( np.dot( Rz, Ry ), Rx )
        # rotate point cloud
        self.pc = np.vstack( [ self.pc.T, np.ones( self.pc.shape[0] ) ] ) # append ones
        self.pc =  np.dot( R, self.pc ) # rotate -> pc_rot = R*X
        self.pc = self.pc[0:3,:].T # remove ones

    def __RotationX( self, angle_ ):
        """ 3d rotation about the x axis
            - input:
                angle_: angle of rotation in radians
            - output:
            Rx: rotation matrix (no translation)
        """

        Rx = np.array( [ [ 1., 0., 0., 0. ],
                [ 0., np.cos( angle_ ), -np.sin( angle_ ), 0. ],
                [ 0., np.sin( angle_ ), np.cos( angle_ ), 0. ],
                [ 0., 0., 0., 1. ]
                ] )
        return Rx

    def __RotationY( self, angle_ ):
        """ 3d rotation about the y axis
            - input:
                angle_: angle of rotation in radians
            - output:
            Ry: rotation matrix (no translation)
        """
        Ry = np.array( [ [ np.cos( angle_ ), 0., np.sin( angle_ ), 0. ],
                [ 0., 1., 0., 0. ],
                [ -np.sin( angle_ ), 0., np.cos( angle_ ), 0. ],
                [ 0., 0., 0., 1. ]
                ] )
        return Ry

    def __RotationZ( self, angle_ ):
        """ 3d rotation about the z axis
            - input:
                angle_: angle of rotation in radians
            - output:
            Rx: rotation matrix (no translation)
        """
        Rz = np.array( [ [ np.cos( angle_ ), -np.sin( angle_ ), 0., 0. ],
                [ np.sin( angle_ ), np.cos( angle_ ), 0., 0. ],
                [ 0., 0., 1., 0. ],
                [ 0., 0., 0., 1. ]
                ] )
        return Rz

    def augmentation_Translation( self ):
        """ Augment the input point cloud by randomly translating it
            - input:
                pc_: point cloud Mx3 numpy matrix
            - output:
                ---: augmentations are applied in place
        """
        pass


    def augmentation_Flipping( self ):
        """ Augment the input point cloud by randomly flipping it
            - input:
                pc_: point cloud Mx3 numpy matrix
            - output:
                ---: augmentations are applied in place
        """
        selectFlip = np.int( np.random.rand(1)[0]*4. )
        # 0 - nothing
        if selectFlip == 1: # about y axis
            self.pc[:,0] *= -1.
        elif selectFlip == 2: # about x axis
            self.pc[:,1] *= -1.
        elif selectFlip == 3: # about x and y axis
            self.pc[:,[0,1]] *= -1.

    def augmentation_Shuffle( self ):

        temp = np.hstack((self.pc, self.pc_labels[:,np.newaxis]))
        np.random.shuffle(temp)
        self.pc = temp[:,:3]
        self.pc_labels = temp[:,3]


    def normalisation( self ):

        l = self.pc.shape[0]
        centroid = np.mean(self.pc, axis=0)
        self.pc = self.pc - centroid
        m = np.max(np.sqrt(np.sum(self.pc ** 2, axis=1)))
        self.pc = self.pc / m

    def vertical_density( self , support_window=1, weighted=True ):

        if support_window==1:
            self.bev_verticalDensity = np.sum(self.og, axis=2) / np.shape(self.og)[2]
        else:
            self.bev_verticalDensity = np.sum(self.og, axis=2)
            if weighted is True:
                g = gaussian_kernel(support_window, mu=0.0, sigma=1.0)
                self.bev_verticalDensity = signal.convolve2d(self.bev_verticalDensity,g,mode='same')
                self.bev_verticalDensity = self.bev_verticalDensity / (np.shape(self.og)[2] * np.sum(g))
            else:
                self.bev_verticalDensity = signal.convolve2d(self.bev_verticalDensity,np.ones((support_window,support_window)),mode='same')
                self.bev_verticalDensity = self.bev_verticalDensity / (np.shape(self.og)[2]*(support_window**2))

    

    def max_height( self, support_window=1, weighted=True ):

        if support_window==1: #old way
            self.bev_maxHeight = np.zeros((np.shape(self.og)[0],np.shape(self.og)[1]))
            for i in range(np.shape(self.og)[0]):
                for j in range(np.shape(self.og)[1]):
                    if np.sum(self.og[i,j,:]>0) > 0:
                        self.bev_maxHeight[i,j] = np.where(self.og[i,j,:]>0)[0][0]
        else:   #new way
            self.bev_maxHeight = np.zeros((np.shape(self.og)[0], np.shape(self.og)[1]))
            non_empties = np.where(self.og > 0)
            self.bev_maxHeight[non_empties[0], non_empties[1]] = non_empties[2]
            if weighted is True: # dont do these for support_window==1
                g = gaussian_kernel(support_window, mu=0.0, sigma=1.0)
                self.bev_maxHeight = ndimage.maximum_filter(self.bev_maxHeight, footprint=g>0.6)
            else:
                self.bev_maxHeight = ndimage.maximum_filter(self.bev_maxHeight, footprint=np.ones((support_window,support_window)))
        

    def mean_returns( self, support_window=1, weighted=True ):

        if support_window==1:
            self.bev_meanReturn = np.mean(self.og_returns, axis=2)
        else:
            self.bev_meanReturn = np.sum(self.og_returns, axis=2)
            if weighted is True:
                g = gaussian_kernel(support_window, mu=0.0, sigma=1.0)
                self.bev_meanReturn = signal.convolve2d(self.bev_meanReturn,g,mode='same')
                self.bev_meanReturn = self.bev_meanReturn / (np.shape(self.og)[2] * np.sum(g))
            else:
                self.bev_meanReturn = signal.convolve2d(self.bev_meanReturn,np.ones((support_window, support_window)),mode='same')
                self.bev_meanReturn = self.bev_meanReturn / (np.shape(self.og)[2] * (support_window ** 2))


    def max_returns( self, support_window=1, weighted=True ):

        self.bev_maxReturn = np.max(self.og_returns, axis=2)
        if support_window>1:
            if weighted is True:
                g = gaussian_kernel(support_window, mu=0.0, sigma=1.0)
                self.bev_maxReturn = signal.convolve2d(self.bev_maxReturn,g,mode='same')
            else:
                self.bev_maxReturn = signal.convolve2d(self.bev_maxReturn,np.ones((support_window, support_window)),mode='same')


    def rasterise( self, res_=0.1, gridSize_=np.array( (32,32) ) ):
        """ Convert point cloud to 2D raster representation
            Assumes point cloud has been recentred
            - input:
                   pc_: point cloud Mx3 numpy matrix
                   res_: resolution of each cell in metres
                   gridSize_: size of the occupancy grid (DEFAULT: 32x32)
            - output:
              ---: 2D raster is a class variable
        """
        # initialise grid
        self.raster = np.zeros( gridSize_ )
        # recentre the point cloud about the mean if it is not done already
        if not self.__flag_recentred:
            c = np.mean( self.pc, axis=0 )
            self.recentre( c  )

        # get index of points within grid
        if isinstance(res_,(list)): # possible to specify different res for each dimension
            self.__index_x = np.array(np.clip(np.floor((self.pc[:, 0] - (-gridSize_[0] / 2. * res_[0])) / res_[0]), 0, gridSize_[0] - 1), dtype=int)
            self.__index_y = np.array(np.clip(np.floor((self.pc[:, 1] - (-gridSize_[1] / 2. * res_[1])) / res_[1]), 0, gridSize_[1] - 1), dtype=int)
        else:
            self.__index_x = np.array( np.clip( np.floor( ( self.pc[:,0]-(-gridSize_[0]/2.*res_) )/res_ ), 0, gridSize_[0]-1 ), dtype=int)
            self.__index_y = np.array( np.clip( np.floor( ( self.pc[:,1]-(-gridSize_[1]/2.*res_) )/res_ ), 0, gridSize_[1]-1 ), dtype=int)
        # set cells to occupied
        self.raster[self.__index_x,self.__index_y] = 1.


    def pcd2rasterCoords(self,pts,gridSize,res):
        # make sure to pass in pts with shape [numSamples x numCoordinates]

        if np.shape(res) == ():
            res = np.tile(res,(np.shape(pts)[1]))

        centred_pts = pts - self.__centre[:np.shape(pts)[1]]

        # note: x point corresponds to row (first index) in the og grid -> therefore y (and vice-versa for y)
        coords = {}
        coords['row'] = np.array( np.clip( np.floor( ( centred_pts[:,0]-(-gridSize[0]/2.*res[0]) )/res[0] ), 0, gridSize[0]-1 ), dtype=int) #y
        coords['col'] = np.array( np.clip( np.floor( ( centred_pts[:,1]-(-gridSize[1]/2.*res[1]) )/res[1] ), 0, gridSize[1]-1 ), dtype=int) #x
        if np.shape(pts)[1] == 3:
            coords['z'] = np.array(np.clip(np.floor((centred_pts[:, 2] - (-gridSize[2] / 2. * res[2])) / res[2]), 0, gridSize[2] - 1),dtype=int)
        return coords
    
    
    def ground_normalise(self,ground_pts ):
        xmax = np.max(self.pc[:, 0])
        xmin = np.min(self.pc[:, 0])
        ymax = np.max(self.pc[:, 1])
        ymin = np.min(self.pc[:, 1])

        idx = (ground_pts[:, 0] > xmin) & (ground_pts[:, 0] < xmax) & (ground_pts[:, 1] > ymin) & (ground_pts[:, 1] < ymax)
        ground_pts_cropped = ground_pts[idx, :]
        if np.sum(idx)==0:
            ground_pts_cropped = ground_pts

        tree = KDTree(ground_pts_cropped[:,[0,1]])
        dist,ind = tree.query(self.pc[:,[0,1]],k=1)

        self.pc[:,2] -=  ground_pts_cropped[ind[:,0],2]


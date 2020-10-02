import tensorflow as tf
if tf.__version__.split('.')[0]=='2':
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()

import numpy as np
from os.path import join, exists, basename, split
from sklearn.neighbors import KDTree
import sys

from forest3D import network_ops,processLidar,utilities,lidar_IO
from forest3D.voxelSegModels import Voxnet as SegModel


class VoxelStemSeg():


    def __init__(self, input_dims,res,isReturns=False,nClasses=3,mdl_name='segmenter' ):

        self._input_dims = input_dims # [150,150,100]
        self._res = res #[0.1,0.1,0.4]
        self._nClasses = nClasses
        self.isReturns = isReturns
        self._ph_input = tf.placeholder(dtype=tf.float32, shape=[None,1]+input_dims)
        self._mdl_name = mdl_name
        self._ph_keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

        # build graph
        self._segModel = SegModel(self._ph_input,self._ph_keep_prob,self._mdl_name,self._nClasses)
        self._output = self._segModel._output
        self._vars = self._segModel._vars



    def train_op(self,opt_method='Adam',lr=1e-3,decay_rate=None,decay_steps=None,piecewise_bounds=None,piecewise_values=None):
        """
        Create necessary nodes to train the model.


        opt_method: optimisation method (string). ['Adam','RMS','SGD']
        lr: learning rate (float).
        decay_rate : rate of decay of learning rate (float). None -> no decay
        decay_steps: epochs to decay the learning rate (list of integers). None -> no decay
        piecewise_bounds:
        piecewise_values:
        """

        self._ph_target = tf.placeholder(dtype=tf.float32, shape=[None,self._nClasses]+self._input_dims)

        # loss node
        self._class_weights = network_ops.balance_classes_3d(self._ph_target,num_classes=self._nClasses)

        self._loss = network_ops.losses_3d( self._output,self._ph_target,class_weights=self._class_weights )

        # optimise node
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self._optimiser = network_ops.train_step(self._loss, learning_rate=lr, decay_rate=decay_rate,
                                                     decay_steps=decay_steps,piecewise_bounds=piecewise_bounds,
                                                     piecewise_values=piecewise_values, method=opt_method,var_list=self._vars)



    def train_model(self,train_iter, val_iter, train_bs, val_bs, save_addr, nEpochs, save_epochs=[],keep_prob=1.0,
                    save_figure=1,augment=False,numAugs=3):
        """
        Trains the model using iterators for the train and val data.



        """

        nSamples = train_iter.data_num

        random_train_idx = np.random.permutation(nSamples)[:train_bs] # train_bs
        random_val_idx = np.random.permutation(val_iter.data_num)[:val_bs]


        if nEpochs not in save_epochs:
            save_epochs.append( nEpochs )


        with tf.Session() as sess:


            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(self._vars)


            # check if addr has 'epoch' in name and contains a checkpoint
            if exists(join(save_addr, 'checkpoint')) & ('epoch' in basename(save_addr)):
                # load a checkpoint
                saver.restore(sess, join(save_addr, 'model.ckpt'))
                epoch_start = int((basename(save_addr)).split('_')[-1]) + 1
                save_addr = split(save_addr)[0]
            else:
                # save directory is empty
                epoch_start = 0


            plot_losses = utilities.plot_vars(['train_loss', 'val_loss'], join(save_addr, 'losses.png'))

            # iterate over batches
            numIters = int(np.ceil(nSamples/float(train_bs)))
            for epoch in range(epoch_start,nEpochs+1):

                for iteration in range(numIters):

                    print('epoch: %i/%i, iteration: %i/%i' % (epoch, nEpochs, iteration,numIters))

                    if self.isReturns == True:
                        _, train_labels, train_batch = train_iter.next_batch2(augment=augment, numAugs=numAugs, angle_x_randLim=0,
                                                                                 angle_y_randLim=0)  # returns pcd,labels,returns
                    else:
                        train_batch, train_labels = train_iter.next_batch2(augment=augment, numAugs=numAugs, angle_x_randLim=0,
                                                                                 angle_y_randLim=0)  # returns pcd,labels,returns

                    feed = {self._ph_input: train_batch, self._ph_target: train_labels,self._ph_keep_prob: keep_prob}
                    sess.run(self._optimiser, feed)


                train_iter.reset_batch()

                # compute losses
                if epoch % save_figure == 0:

                    if self.isReturns == True:
                        _, train_labels, train_batch = train_iter.get_pc(list(random_train_idx), augment=False)
                        _, val_labels, val_batch = val_iter.get_pc(list(random_val_idx), augment=False)
                    else:
                        train_batch, train_labels = train_iter.get_pc(list(random_train_idx), augment=False)
                        val_batch, val_labels = val_iter.get_pc(list(random_val_idx), augment=False)

                    feed = {self._ph_input: train_batch, self._ph_target: train_labels,self._ph_keep_prob: 1.0}
                    train_loss = sess.run(self._loss, feed)

                    feed = {self._ph_input: val_batch, self._ph_target: val_labels,self._ph_keep_prob: 1.0}
                    val_loss = sess.run(self._loss, feed)

                    plot_losses.update_plot(epoch, [train_loss, val_loss])

                # save model
                if epoch in save_epochs:
                    saver.save(sess, join(save_addr, "epoch_%i" % (epoch), "model.ckpt"))


    def predict(self,xyzr_list,model_addr,returnSplit=False,batchsize=5,dist_threshold=0.5):
        '''

        :param model_addr:
        :param xyzr_list_fold. list(np.array): list of pointcloud arrays. Each array is nPoints x 4. Columns are x,y,z,return.
        :return:
        '''
        numTrees = len(xyzr_list)
        #batchsize = [i for i in range(1, 10) if numTrees % i is 0][-1]
        batchsize = min(numTrees, batchsize)

        if self.isReturns == True:
            assert(np.shape(xyzr_list[0])[1] == 4)

        xyz_list = []
        for i in range(numTrees):
            xyz_list.append(xyzr_list[i][:, :3])


        if self.isReturns is True:
            return_list = []
            for i in range(numTrees):
                return_list.append(xyzr_list[i][:, 3])
        else:
            return_list = None

        iter = processLidar.iterator_binaryVoxels_pointlabels(xyz_list, returns=return_list, res=self._res,
                                                              gridSize=self._input_dims,
                                                              numClasses=self._nClasses - 1,
                                                              batchsize=batchsize)

        with tf.Session() as sess:

            saver = tf.train.Saver(self._vars)
            saver.restore(sess, join(model_addr, 'model.ckpt'))

            pred2 = np.zeros(([numTrees] + self._input_dims))
            ogOffsets = np.zeros(([numTrees, 3]))
            remainder = numTrees % batchsize
            if remainder>0:
                numBatches = int(numTrees / batchsize)+1
            else:
                numBatches = int(numTrees / batchsize)
            for i in range(numTrees // batchsize):
                sys.stdout.write("\rinference step: %d out of %d batches" % (i+1, numBatches))
                sys.stdout.flush()
                if self.isReturns == True:
                    _, data_x, ogOffsets_i = iter.next_batch(augment=False, outputOffset=True)
                else:
                    data_x, ogOffsets_i = iter.next_batch(augment=False, outputOffset=True)
                pred = sess.run(self._output, feed_dict={self._ph_input: data_x, self._ph_keep_prob: 1.0})
                pred2[i * batchsize:(i + 1) * batchsize, ...] = np.argmax(pred, axis=1)
                ogOffsets[i * batchsize:(i + 1) * batchsize, ...] = ogOffsets_i
            iter.reset_batch()
            if remainder>0:
                sys.stdout.write("\rinference step: %d out of %d batches" % (i+2, numBatches))
                sys.stdout.flush()
                if self.isReturns == True:
                    _, data_x, ogOffsets_i = iter.get_pc(range(numTrees)[-remainder:], outputOffset=True)
                else:
                    data_x, ogOffsets_i = iter.get_pc(range(numTrees)[-remainder:], outputOffset=True)
                pred = sess.run(self._output, feed_dict={self._ph_input: data_x, self._ph_keep_prob: 1.0})
                pred2[-remainder:, ...] = np.argmax(pred, axis=1)
                ogOffsets[-remainder:, ...] = ogOffsets_i
            sys.stdout.write("\n")

        pcd_pred = []
        for i in range(np.shape(pred2)[0]):
            sys.stdout.write("\rconvert voxels to pointclouds step: %d out of %d trees" % (i + 1, np.shape(pred2)[0]))
            sys.stdout.flush()
            pcd_pred.append(
                lidar_IO.WriteOG(pred2[i, ...], res=self._res, offset=[0, 0, 0], ogOffset=ogOffsets[i, :],
                                 returnPcd=True, labels=True))
        sys.stdout.write("\n")

        pcd_list_labels = []
        for i in range(len(xyz_list)):
            sys.stdout.write("\rkd-tree step: %d out of %d trees" % (i+1, len(xyz_list)))
            sys.stdout.flush()
            kdtree = KDTree(pcd_pred[i][:, :3])  # build kdtree from downsampled pcd with predicted labels
            [dist, idx] = kdtree.query(xyz_list[i])  # 0.5 query nearest downsampled tree points with points from original tree
            idx = idx[:, 0]
            dist = dist[:, 0]
            tmp = np.zeros((np.shape(xyz_list[i])[0]))
            tmp[dist < dist_threshold] = pcd_pred[i][(idx)[dist < dist_threshold], 3]  # extract label of nearest point in downsampled tree. 1000 represent inf
            pcd_list_labels.append(tmp)
            # to remove unknown points
            xyz_list[i] = xyz_list[i][dist < dist_threshold, :]
            pcd_list_labels[i] = pcd_list_labels[i][dist < dist_threshold]
        sys.stdout.write("\n")


        if returnSplit:
            return xyz_list,pcd_list_labels
        else:
            labelled_list = []
            for i in range(numTrees):
                labelled_list.append( np.hstack(( xyz_list[i], pcd_list_labels[i][:,np.newaxis] )) )

            return labelled_list



    def predict_uncertainty(self,xyzr_list,model_addr,n_passes=5,keep_prob=0.6,batchsize=5,returnSplit=False,dist_threshold=0.5):
        '''

        :param model_addr:
        :param xyzr_list_fold. list(np.array): list of pointcloud arrays. Each array is nPoints x 4. Columns are x,y,z,return.
        :return:
        '''
        numTrees = len(xyzr_list)
        #batchsize = [i for i in range(1, 10) if numTrees % i is 0][-1]
        batchsize = min(numTrees,batchsize)

        if self.isReturns == True:
            assert(np.shape(xyzr_list[0])[1] == 4)

        xyz_list = []
        for i in range(numTrees):
            xyz_list.append(xyzr_list[i][:, :3])


        if self.isReturns is True:
            return_list = []
            for i in range(numTrees):
                return_list.append(xyzr_list[i][:, 3])
        else:
            return_list = None

        iter = processLidar.iterator_binaryVoxels_pointlabels(xyz_list, returns=return_list, res=self._res,
                                                              gridSize=self._input_dims,
                                                              numClasses=self._nClasses - 1,
                                                              batchsize=batchsize)

        with tf.Session() as sess:

            saver = tf.train.Saver(self._vars)
            saver.restore(sess, join(model_addr, 'model.ckpt'))


            data_size = [n_passes, numTrees, self._nClasses] + self._input_dims
            mc_scores = np.zeros((data_size))
            remainder = numTrees % batchsize
            ogOffsets = np.zeros(([numTrees, 3]))
            for i in range(n_passes):
                sys.stdout.write("\rinference step: %d out of %d passes" % (i + 1, n_passes))
                sys.stdout.flush()
                for j in range(numTrees // batchsize):
                    if self.isReturns == True:
                        _, data_x, ogOffsets_i = iter.next_batch(augment=False, outputOffset=True)
                    else:
                        data_x, ogOffsets_i = iter.next_batch(augment=False, outputOffset=True)
                    feed = {self._ph_input: data_x, self._ph_keep_prob: keep_prob}
                    mc_scores[i, j * batchsize:(j + 1) * batchsize, :] = sess.run(tf.nn.softmax(self._output, dim=1),feed)
                    ogOffsets[j * batchsize:(j + 1) * batchsize, ...] = ogOffsets_i
                iter.reset_batch()
                if remainder > 0:
                    if self.isReturns == True:
                        _, data_x, ogOffsets_i = iter.get_pc(range(numTrees)[-remainder:], outputOffset=True)
                    else:
                        data_x, ogOffsets_i = iter.get_pc(range(numTrees)[-remainder:], outputOffset=True)
                    feed = {self._ph_input: data_x, self._ph_keep_prob: keep_prob}
                    mc_scores[i, -remainder:, :] = sess.run(tf.nn.softmax(self._output, dim=1), feed)
                    ogOffsets[-remainder:, ...] = ogOffsets_i
            mc_uncertainty = (np.mean(mc_scores, axis=0), np.std(mc_scores, axis=0) ** 2)  # mean and variance for each class prediction
            class_uncertainty = np.max(mc_uncertainty[0], axis=1)
            class_label = np.argmax(mc_uncertainty[0], axis=1)
            sys.stdout.write("\n")


        pcd_pred = []
        for i in range(np.shape(class_label)[0]):
            sys.stdout.write("\rconvert voxels to pointclouds step: %d out of %d trees" % (i + 1, np.shape(class_label)[0]))
            sys.stdout.flush()
            tmp_labels = lidar_IO.og2xyz(class_label[i, ...], class_label[i, ...] > 0, res=self._res,ogOffset=ogOffsets[i, :])
            tmp_probs = lidar_IO.og2xyz(class_uncertainty[i, ...], class_label[i, ...] > 0, res=self._res,ogOffset=ogOffsets[i, :])
            pcd_pred.append(np.hstack((tmp_probs, tmp_labels[:, 3][:, np.newaxis])))
        sys.stdout.write("\n")

        pcd_list_labels = []
        for i in range(numTrees):
            sys.stdout.write("\rkd-tree step: %d out of %d trees" % (i + 1,numTrees))
            sys.stdout.flush()
            kdtree = KDTree(pcd_pred[i][:, :3])  # build kdtree from downsampled pcd with predicted labels
            [dist, idx] = kdtree.query(xyz_list[i])  # 0.5 query nearest downsampled tree points with points from original tree
            idx = idx[:, 0]
            dist = dist[:, 0]
            tmp = pcd_pred[i][(idx)[dist < dist_threshold],3:5]  # extract label of nearest point in downsampled tree. 1000 represent inf
            pcd_list_labels.append(tmp)
            # to remove unknown points
            xyz_list[i] = xyz_list[i][dist < dist_threshold, :]


        if returnSplit:
            return xyz_list,pcd_list_labels
        else:
            labelled_list = []
            for i in range(numTrees):
                labelled_list.append( np.hstack(( xyz_list[i], pcd_list_labels[i] )) )

            return labelled_list
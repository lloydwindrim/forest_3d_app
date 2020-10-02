import tensorflow as tf
if tf.__version__.split('.')[0]=='2':
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()

import numpy as np
from os.path import join, exists, basename, split
from sklearn.neighbors import KDTree
import sys

from forest3D import network_ops,processLidar,utilities,lidar_IO
from forest3D.voxelVaeModels import Voxnet_encoder,Voxnet_decoder


class VoxelLatentEncoder():


    def __init__(self, input_dims,res,isReturns=False,latent_dims=2,mdl_name='vae',mode='train' ):

        self._input_dims = input_dims # [150,150,100]
        self._res = res #[0.1,0.1,0.4]
        self._latent_dims = latent_dims
        self.isReturns = isReturns
        self._ph_input = tf.placeholder(dtype=tf.float32, shape=[None,1]+input_dims)
        self._mdl_name = mdl_name
        self._ph_keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        self._ph_is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        # build graph
        if mode == 'train':
            self._build_train_graph()
        elif mode == 'encode':
            self._build_encoder_graph()
        elif mode == 'decode':
            self._build_decoder_graph()



    def _build_train_graph(self):

        # build encoder
        self._encoder = Voxnet_encoder(self._ph_input, self._ph_keep_prob, self._ph_is_training, self._mdl_name,
                                       self._latent_dims)

        self._code_random = self._encoder.get_code('sample')
        self._code = self._encoder.get_code('mean')


        # build decoder
        self._decoder_loss = Voxnet_decoder(self._code_random, self._input_dims,self._ph_keep_prob, self._ph_is_training,
                                            self._mdl_name)

    def _build_encoder_graph(self):

        self._encoder = Voxnet_encoder(self._ph_input, self._ph_keep_prob, self._ph_is_training, self._mdl_name,
                                       self._latent_dims)

        self._code = self._encoder.get_code('mean')



    def _build_decoder_graph(self):

        # build encoder
        self._encoder = Voxnet_encoder(self._ph_input, self._ph_keep_prob, self._ph_is_training, self._mdl_name,
                                       self._latent_dims)

        self._code = self._encoder.get_code('mean')


        # build decoder
        self._decoder_gen = Voxnet_decoder(self._code, self._input_dims,self._ph_keep_prob, self._ph_is_training,
                                            self._mdl_name)

        self._reconstruction = self._decoder_gen.get_output('mean')



    def train_op(self, opt_method='Adam', lr=1e-3, alpha=1.0, decay_rate=None, decay_steps=None, piecewise_bounds=None,
                 piecewise_values=None):
        """
        Create necessary nodes to train the model.


        opt_method: optimisation method (string). ['Adam','RMS','SGD']
        lr: learning rate (float).
        decay_rate : rate of decay of learning rate (float). None -> no decay
        decay_steps: epochs to decay the learning rate (list of integers). None -> no decay
        piecewise_bounds:
        piecewise_values:
        """

        self._posterior = self._encoder.get_posterior()
        self._prior = network_ops.make_prior(code_size=self._latent_dims)
        self._likelihood = self._decoder_loss.get_likelihood(self._ph_input)

        self._alpha = alpha
        self._divergence = network_ops.loss_KL_divergence(self._posterior, self._prior)

        # loss node
        self._elbo = tf.reduce_mean(self._likelihood - (self._alpha * self._divergence))

        # optimise node
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self._optimiser = network_ops.train_step(-self._elbo, learning_rate=lr, decay_rate=decay_rate,
                                                     decay_steps=decay_steps, piecewise_bounds=piecewise_bounds,
                                                     piecewise_values=piecewise_values, method=opt_method,
                                                     var_list=[
                                                         self._encoder.get_vars() + self._decoder_loss.get_vars()])



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
            saver = tf.train.Saver()


            # check if addr has 'epoch' in name and contains a checkpoint
            if exists(join(save_addr, 'checkpoint')) & ('epoch' in basename(save_addr)):
                # load a checkpoint
                saver.restore(sess, join(save_addr, 'model.ckpt'))
                epoch_start = int((basename(save_addr)).split('_')[-1]) + 1
                save_addr = split(save_addr)[0]
            else:
                # save directory is empty
                epoch_start = 0

            plot_likelihood = utilities.plot_vars(['train_likelihood', 'val_likelihood'],join(save_addr, 'likelihood.png'))
            plot_divergence = utilities.plot_vars(['train_divergence', 'val_divergence'],join(save_addr, 'divergence.png'))

            # iterate over batches
            numIters = int(np.ceil(nSamples/float(train_bs)))
            for epoch in range(epoch_start,nEpochs+1):

                for iteration in range(numIters):

                    print('epoch: %i/%i, iteration: %i/%i' % (epoch, nEpochs, iteration,numIters))

                    if self.isReturns == True:
                        _, train_batch = train_iter.next_batch2(augment=augment, numAugs=numAugs, angle_x_randLim=0,
                                                                                 angle_y_randLim=0)  # returns pcd,labels,returns
                    else:
                        train_batch = train_iter.next_batch2(augment=augment, numAugs=numAugs, angle_x_randLim=0,
                                                                                 angle_y_randLim=0)  # returns pcd,labels,returns

                    feed = {self._ph_input: train_batch, self._ph_keep_prob: keep_prob, self._ph_is_training: True}
                    sess.run(self._optimiser, feed)


                train_iter.reset_batch()

                # compute losses
                if epoch % save_figure == 0:

                    if self.isReturns == True:
                        _, train_batch = train_iter.get_pc(list(random_train_idx), augment=False)
                        _, val_batch = val_iter.get_pc(list(random_val_idx), augment=False)
                    else:
                        train_batch = train_iter.get_pc(list(random_train_idx), augment=False)
                        val_batch = val_iter.get_pc(list(random_val_idx), augment=False)

                    feed = {self._ph_input: train_batch, self._ph_keep_prob: 1.0, self._ph_is_training: False}
                    train_likelihood, train_divergence = sess.run([self._likelihood, self._divergence], feed)

                    feed = {self._ph_input: val_batch, self._ph_keep_prob: 1.0, self._ph_is_training: False}
                    val_likelihood, val_divergence = sess.run([self._likelihood, self._divergence], feed)

                    plot_likelihood.update_plot(epoch, [np.mean(train_likelihood), np.mean(val_likelihood)])
                    plot_divergence.update_plot(epoch, [np.mean(train_divergence), np.mean(val_divergence)])

                # save model
                if epoch in save_epochs:
                    saver.save(sess, join(save_addr, "epoch_%i" % (epoch), "model.ckpt"))


    def encode_input(self,data_iter,model_addr):
        '''

        :param data_iter: Assumes a return, points iterator.
        :param model_addr:
        :return:
        '''

        with tf.Session() as sess:

            if (self._deploy == 'decoder') | (self._deploy == 'encoder') | (self._deploy == None):
                saver = tf.train.Saver(self._encoder.get_vars())
            else:
                raise ValueError("deploy mode must be 'encoder' or None")
            saver.restore(sess, join(model_addr, 'model.ckpt'))

            codes = np.zeros((data_iter.data_num, self._codeSize))
            remainder = data_iter.data_num % data_iter.batchsize


            for j in range(data_iter.data_num // data_iter.batchsize):
                _,samples = data_iter.next_batch()
                feed = {self._ph_input: samples, self._ph_keep_prob: 1.0, self._ph_is_training: False}
                codes[j * data_iter.batchsize:(j + 1) * data_iter.batchsize, :] = sess.run(self._code,feed)

            if remainder > 0:
                _,samples = data_iter.get_pc(range(data_iter.data_num)[-remainder:])
                feed = {self._ph_input: samples, self._ph_keep_prob: 1.0, self._ph_is_training: False}
                codes[-remainder:, :] = sess.run(self._code,feed)

            return codes

    def reconstruct_input(self,data_iter,model_addr):
        '''

        :param data_iter: Assumes a return, points iterator.
        :param model_addr:
        :return:
        '''

        with tf.Session() as sess:

            if (self._deploy == 'decoder') | (self._deploy == None):
                saver = tf.train.Saver(self._encoder.get_vars()+self._decoder_gen.get_vars())
            else:
                raise ValueError("deploy mode must be 'decoder' or None")
            saver.restore(sess, join(model_addr, 'model.ckpt'))

            recons = np.zeros((data_iter.data_num, 1, self._input_dims[0], self._input_dims[1], self._input_dims[2]))
            remainder = data_iter.data_num % data_iter.batchsize


            for j in range(data_iter.data_num // data_iter.batchsize):
                _,samples = data_iter.next_batch()
                feed = {self._ph_input: samples, self._ph_keep_prob: 1.0, self._ph_is_training: False}
                recons[j * data_iter.batchsize:(j + 1) * data_iter.batchsize, ...] = sess.run(self._reconstruction,feed)

            if remainder > 0:
                _,samples = data_iter.get_pc(range(data_iter.data_num)[-remainder:])
                feed = {self._ph_input: samples, self._ph_keep_prob: 1.0, self._ph_is_training: False}
                recons[-remainder:, ...] = sess.run(self._reconstruction,feed)

            return recons
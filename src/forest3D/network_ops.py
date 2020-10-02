'''
    Description: various functions for deep learning built on-top of tensorflow. The high-level modules in the package
    call these functions.

    - File name: network_ops.py
    - Author: Lloyd Windrim
    - Date created: June 2019
    - Python package: deephyp

'''
import tensorflow as tf
if tf.__version__.split('.')[0]=='2':
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
    tfd = tf.distributions
else:
    tfd = tf.contrib.distributions

import math
import numpy as np
from os.path import join, exists, basename, split




def create_variable(shape,method='gaussian',wd=False):
    """ Setup a trainable variable (collection of parameters) of a particular shape.

    Args:
        shape (list): Data shape.
        method (str): How to initialise parameter values.
        wd (boolean): Setup weight decay for this variable.

    Returns:
        (tensor): Set of parameters for the given variable.
    """
    return tf.Variable(init_weight(method, shape, wd=wd))


def layer_fullyConn(input, W, b):
    """ Define a fully connected layer operation. Also called a 'dense' layer.

    Args:
        input (tensor): Data input into the layer. Shape [numSamples x numInputNeurons].
        W (tensor): Weight parameters for the layer. Shape [numInputNeurons x numOutputNeurons].
        b (tensor): Bias parameters for the layer. Shape [numOutputNeurons].

    Returns:
        (tensor): Computes layer output. Shape [numSamples x numOutputNeurons].
    """
    return tf.matmul(input, W) + b

def layer_conv1d(input, W, b, stride=1,padding='SAME'):
    """ Define a 1 dimensional convolution layer operation.

    Args:
        input (tensor): Data input into the layer. Shape [numSamples x numInputNeurons x numFiltersIn].
        W (tensor): Weight parameters of the filters/kernels. Shape [filterSize x numFiltersIn x numFiltersOut].
        b (tensor): Bias parameters for the layer. Shape [numFiltersOut].
        stride (int): Stride at which to convolve (must be >= 1).
        padding (str): Type of padding to use ('SAME' or 'VALID').

    Returns:
        (tensor): Computes layer output. Shape [numSamples x numOutputNeurons x numFiltersOut].
    """

    if (padding!='SAME')&(padding!='VALID'):
        raise ValueError('unknown padding type: %s. Use SAME or VALID' % padding)
    if stride < 1:
        raise ValueError('stride must be greater than 0. Stride = %d found in conv layer.'% stride)

    return tf.nn.conv1d(input,W,stride=stride,padding=padding) + b


def layer_deconv1d(input, W, b, outputShape, stride=1,padding='SAME'):
    """ Define a 1 dimensional deconvolution layer operation. Also called convolutional transpose or upsampling layer.

    Args:
        input (tensor): Data input into the layer. Shape [numSamples x numInputNeurons x numFiltersIn].
        W (tensor): Weight parameters of the filters/kernels. Shape [filterSize x numFiltersOut x numFiltersIn].
        b (tensor): Bias parameters for the layer. Shape [numFiltersOut].
        outputShape (list): Expected shape of the layer output. Shape [numSamples x numOutputNeurons x numFiltersOut].
        stride (int): Stride at which to convolve (must be >= 1).
        padding (str): Type of padding to use ('SAME' or 'VALID').

    Returns:
        (tensor): Computes layer output. Shape [numSamples x numOutputNeurons x numFiltersOut].
    """

    if (padding!='SAME')&(padding!='VALID'):
        raise ValueError('unknown padding type: %s. Use SAME or VALID' % padding)
    if stride < 1:
        raise ValueError('stride must be greater than 0. Stride = %d found in deconv layer.'% stride)

    return tf.nn.conv1d_transpose(input,W,outputShape,strides=stride,padding=padding) + b




def conv_output_shape(inputShape, filterSize, padding, stride):
    """ Computes the expected output shape (for the convolving axis only) of a convolution layer given an input shape.

    Args:
        inputShape (int): Shape of convolving axis of input data.
        filterSize (int): Size of filter/kernel of convolution layer.
        stride (int): Stride at which to convolve (must be >= 1).
        padding (str): Type of padding to use ('SAME' or 'VALID').

    Returns:
        (int): Output shape of convolving axis for given layer and input shape.
    """
    if padding=='VALID':
        outputShape = np.ceil( (inputShape - (filterSize-1))/stride )
    elif padding=='SAME':
        outputShape = np.ceil(inputShape / stride)
    else:
        raise ValueError('unknown padding type: %s. Use SAME or VALID' % padding)

    return int(outputShape)


def build_conv_layer(ph_input, kernel_size, filters, strides, activation, ph_keep_prob, useBN=False,
                     ph_is_training=False, momentum=0.99,padding='same',dims=2,data_format='channels_last'):

    if dims == 2:
        conv = tf.layers.conv2d(ph_input, kernel_size=kernel_size, filters=filters, strides=strides, padding=padding,
                                activation=activation)
    elif dims == 1:
        conv = tf.layers.conv1d(ph_input, kernel_size=kernel_size, filters=filters, strides=strides, padding=padding,
                                activation=activation)
    elif dims == 3:
        conv = tf.layers.conv3d(ph_input, kernel_size=kernel_size, filters=filters, strides=strides, padding=padding,
                                activation=activation, use_bias=False, data_format=data_format)
    dropout = tf.nn.dropout(conv, ph_keep_prob)
    if useBN == True:
        return tf.contrib.layers.batch_norm(dropout, is_training=ph_is_training, decay=momentum)
    else:
        return dropout


def build_deconv_layer(ph_input, kernel_size, filters, strides, activation, ph_keep_prob, useBN=False,
                     ph_is_training=False, momentum=0.99, padding='same',dims=2,data_format='channels_last'):

    if dims==2:
        conv = tf.layers.conv2d_transpose(ph_input, kernel_size=kernel_size, filters=filters, strides=strides, padding=padding,
                            activation=activation)
    elif dims==3:
        conv = tf.layers.conv3d_transpose(ph_input, kernel_size=kernel_size, filters=filters, strides=strides, padding=padding,
                            activation=activation, use_bias=False, data_format=data_format)

    dropout = tf.nn.dropout(conv, ph_keep_prob)
    if useBN == True:
        return tf.contrib.layers.batch_norm(dropout, is_training=ph_is_training, decay=momentum)
    else:
        return dropout


def build_dense_layer(ph_input, units, activation, ph_keep_prob, useBN=False, ph_is_training=False, momentum=0.99):

    dense = tf.layers.dense(ph_input, units=units, activation=activation)
    dropout = tf.nn.dropout(dense, ph_keep_prob)
    if useBN == True:
        return tf.contrib.layers.batch_norm(dropout, is_training=ph_is_training, decay=momentum)
    else:
        return dropout


def layer_activation(func='linear'):
    """ Return an activation function operation. Can be passed to tf.layers activation argument.
    Args:
        input (tensor): Data input into the function.
        func (str): Type of activation function. (relu, sigmoid, linear).
    Returns:
        (function): Computes activation. Shape is same as input.
    """

    if func == 'relu':
        a = tf.nn.relu
    elif func == 'sigmoid':
        a = tf.nn.sigmoid
    elif func == 'tanh':
        a = tf.nn.tanh
    elif func == 'lrelu':
        a = tf.nn.leaky_relu
    elif func == 'linear':
        a = None
    else:
        raise ValueError('unknown activation function: %s. Use relu, lrelu, sigmoid or linear.' % func)

    return a


def train_step(loss, learning_rate=1e-3, decay_steps=None, decay_rate=None, piecewise_bounds=None, piecewise_values=None,
             method='Adam', var_list=None):
    """ Operation for training the weights of the network by optimising them to minimise the loss function. Note that \
        the default is a constant learning rate (no decay).

    Args:
        loss (tensor): Output of network loss function.
        learning_rate: (float) Controls the degree to which the weights are updated during training.
        decay_steps (int): Epoch frequency at which to decay the learning rate.
        decay_rate (float): Fraction at which to decay the learning rate.
        piecewise_bounds (int list): Epoch step intervals for decaying the learning rate. Alternative to decay steps.
        piecewise_values (float list): Rate at which to decay the learning rate at the piecewise_bounds.
        method (str): Optimisation method. (Adam, SGD. RMS).

    Returns:
        (op) A train op.
    """


    global_step = tf.Variable(0, trainable=False, name='global_step')

    # update learning rate for current step
    if decay_rate != None:
        lr = tf.train.exponential_decay(learning_rate,
                                        global_step,
                                        decay_steps,
                                        decay_rate, staircase=True)
    elif piecewise_bounds != None:
        lr = tf.train.piecewise_constant(global_step, piecewise_bounds, [learning_rate] + piecewise_values)
    else:
        lr = learning_rate


    if method == 'Adam':
        optimizer = tf.train.AdamOptimizer(lr)
    elif method == 'SGD':
        optimizer = tf.train.GradientDescentOptimizer(lr)
    elif method == 'RMS':
        optimizer = tf.train.RMSPropOptimizer(lr)
    else:
        raise ValueError('unknown optimisation method: %s. Use Adam, SGD or RMS.' % method)

    train_op = optimizer.minimize(loss, global_step=global_step, var_list=var_list)

    return train_op


def loss_function_reconstruction_1D(y_reconstructed,y_target,func='SSE'):
    """ Reconstruction loss function op, comparing 1D tensors for network reconstruction and target.

    Args:
        y_reconstructed (tensor): Output of network (reconstructed 1D vector). Shape [numSamples x inputSize].
        y_target (tensor): What the network is trying to reconstruct (1D vector). Shape [numSamples x inputSize].
        func (string): The name of the loss function to be used. 'SSE'-sum of square errors,'CSA'-cosine spectral angle, \
            'SA'-spectral angle, 'SID'-spectral information divergence.

    Returns:
        (tensor): Reconstruction loss.
    """
    if func == 'SSE':
        # sum of squared errors loss
        loss = tf.reduce_sum( tf.square(y_target - y_reconstructed) )

    elif func == 'CSA':
        # cosine of spectral angle loss
        normalize_r = tf.math.l2_normalize(tf.transpose(y_reconstructed),axis=0)
        normalize_t = tf.math.l2_normalize(tf.transpose(y_target),axis=0)
        loss = tf.reduce_sum( 1 - tf.reduce_sum(tf.multiply(normalize_r, normalize_t),axis=0 ) )

    elif func == 'SA':
        # spectral angle loss
        normalize_r = tf.math.l2_normalize(tf.transpose(y_reconstructed),axis=0)
        normalize_t = tf.math.l2_normalize(tf.transpose(y_target),axis=0)
        loss = tf.reduce_sum( tf.math.acos(tf.reduce_sum(tf.multiply(normalize_r, normalize_t),axis=0 ) ) )

    elif func == 'SID':
        # spectral information divergence loss
        t = tf.divide( tf.transpose(y_target) , tf.reduce_sum(tf.transpose(y_target),axis=0) )
        r = tf.divide( tf.transpose(y_reconstructed) , tf.reduce_sum(tf.transpose(y_reconstructed),axis=0) )
        loss = tf.reduce_sum( tf.reduce_sum( tf.multiply(t,tf.log(tf.divide(t,r))) , axis=0)
                              + tf.reduce_sum( tf.multiply(r,tf.log(tf.divide(r,t))) , axis=0) )
    else:
        raise ValueError('unknown loss function: %s. Use SSE, CSA, SA or SID.' % func)

    return loss


def loss_function_crossentropy_1D( y_pred, y_target, class_weights=None, num_classes=None):
    """ Cross entropy loss function op, comparing 1D tensors for network prediction and target. Weights the classes \
        when calculating the loss to balance un-even training batches. If class weights are not provided, then no \
        weighting is done (weight of 1 assigned to each class).

    Args:
        y_pred (tensor): Output of network (1D vector of class scores). Shape [numSamples x numClasses].
        y_target (tensor): One-hot classification labels (1D vector). Shape [numSamples x numClasses].
        class_weights (tensor): Weight for each class. Shape [numClasses].
        num_classes (int):

    Returns:
        (tensor): Cross-entropy loss.
    """

    if class_weights==None:
        class_weights = tf.constant(1,shape=[y_pred.shape[1]],dtype=tf.float32)

    sample_weights = tf.reduce_sum( tf.multiply(y_target, class_weights ), axis=1) # weight of each sample
    loss = tf.reduce_mean( tf.losses.softmax_cross_entropy(
        onehot_labels=y_target,logits=y_pred,weights=sample_weights ) )

    return loss


def loss_binary_cross_entropy(x, z):
    # x - target [numSamples x 1], z- prediction [numSamples x 1]
    # used to: return loss for each sample [numSamples x 1]
    # now: returns mean loss [1]
    # different to normal softmax cross-entropy: works on a single binary element (not two elements)
    # assumes sigmoid already done
    eps = 1e-12
    return tf.reduce_mean(-(x * tf.log(z + eps) + (1. - x) * tf.log(1. - z + eps)))


def loss_l2_regularisation(var_collection, weighting=1e-6):
    return tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(weighting), var_collection)


def loss_wasserstein(x, z):
    # x - target [numSamples x 1], z- prediction [numSamples x 1]
    # return loss for each sample [1]
    return tf.reduce_mean( ( x * z ) )

def loss_weight_decay(wdLambda):
    """ Weight decay loss op, regularises network by penalising parameters for being too large.

    Args:
        wdLambda (float): Scalar to control weighting of weight decay in loss.

    Returns:
        (tensor) : Weight-decay loss.
    """

    return tf.multiply( wdLambda , tf.reduce_sum(tf.get_collection('wd')) )


def loss_KL_divergence(posterior, prior):

    return tfd.kl_divergence(posterior, prior)

def balance_classes(y_target,num_classes):
    """ Calculates the class weights needed to balance the classes, based on the number of samples of each class in the \
        batch of data.

    Args:
        y_target (tensor): One-hot classification labels (1D vector). Shape [numSamples x numClasses]
        num_classes (int):

    Returns:
        (tensor): A weighting for each class that balances their contribution to the loss. Shape [numClasses].
    """
    y_target = tf.reshape( y_target, [-1, num_classes] )
    class_count = tf.add( tf.reduce_sum( y_target, axis=0 ), tf.constant( [1]*num_classes, dtype=tf.float32 ) )
    class_weights = tf.multiply( tf.divide( tf.ones( ( 1, num_classes) ), class_count ), tf.reduce_max( class_count ) )

    return class_weights


# def make_prior(code_size):
#
#     loc = tf.zeros(code_size)
#     scale = tf.ones(code_size)
#     return tfd.MultivariateNormalDiag(loc, scale)



def class_accuracy(pred,label):

    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(label,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    return accuracy


def accuracy(pred,label):

    correct_prediction = tf.equal(pred,label)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    return accuracy



def save_model(addr,sess,saver,current_epoch,epochs_to_save):
    """Saves a checkpoint at a list of epochs.

    Args:
        addr (str): Address of a directory to save checkpoint for current epoch.
        sess (obj): Tensor flow session object.
        saver (obj): Tensor flow save object.
        current_epoch (int): The current epoch.
        epochs_to_save (int list): Epochs to save checkpoints at.

    """

    if current_epoch in epochs_to_save:
        saver.save(sess, join(addr,"epoch_%i"%(current_epoch),"model.ckpt"))


def load_model(addr,sess):
    """Loads a model from the address of a checkpoint.

    Args:
        addr (str): Address of a directory to save checkpoint for current epoch.
        sess (obj): Tensor flow session object.

    """
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, join(addr, 'model.ckpt'))



def init_weight(opts, shape, stddev=0.1, const=0.1, wd = False, dtype=tf.float32):
    """ Weight initialisation function.

    Args:
        opts (str): Method for initialising variable. ('gaussian','truncated_normal','xavier','xavier_improved', \
            'constant').
        shape (list): Data shape.
        stddev (int): Standard deviation used by 'gaussian' and 'truncated_normal' variable initialisation methods.
        const (int): Constant value to initialise variable to if using 'constant' method.
        wd (boolean): Whether this variable contributes to weight decay or not.
        dtype (tf.dtype): Data type for variable.

    Returns:
        weights:
    """
    if opts == 'gaussian':
        weights = tf.random_normal(shape, stddev=stddev, dtype=dtype)
    elif opts == 'truncated_normal':
        weights = tf.truncated_normal(shape, stddev=stddev)
    elif opts == 'xavier':
        h = shape[0]
        w = shape[1]
        try:
            num_in = shape[2]
        except:
            num_in = 1
        sc = math.sqrt(3.0 / (h * w * num_in))
        weights = tf.multiply(tf.random_normal(shape, dtype=dtype) * 2 - 1, sc)
    elif opts == 'xavier_improved':
        h = shape[0]
        w = shape[1]
        try:
            num_out = shape[3]
        except:
            num_out = 1
        sc = math.sqrt(2.0 / (h * w * num_out))
        weights = tf.multiply(tf.random_normal(shape, dtype=dtype), sc)
    elif opts == 'constant':
        weights = tf.constant(const, shape)
    else:
        raise ValueError('Unknown weight initialization method %s' % opts)

    # set up weight decay on weights
    if wd:
        weight_decay = tf.nn.l2_loss(weights)
        tf.add_to_collection('wd', weight_decay)

    return weights


def balance_classes_3d(y_target,num_classes):
    """ Calculates the class weights needed to balance the classes, based on the number of samples of each class in the \
        batch of data.

        todo: make it work for any num_classes

    Args:
        y_target (tensor): One-hot classification labels (1D vector). Shape [numSamples x numClasses]
        num_classes (int):

    Returns:
        (tensor): A weighting for each class that balances their contribution to the loss. Shape [numClasses].
    """

    y_target = tf.transpose(tf.stack([tf.reshape(y_target[:, 0, ...], [-1]), tf.reshape(y_target[:, 1, ...], [-1]), tf.reshape(y_target[:, 2, ...], [-1])]))
    class_count = tf.add( tf.reduce_sum( y_target, axis=0 ), tf.constant( [1]*num_classes, dtype=tf.float32 ) )
    class_weights = tf.multiply( tf.divide( tf.ones( ( 1, num_classes) ), class_count ), tf.reduce_max( class_count ) )

    return class_weights



def losses_3d(pred,label,class_weights=None):

    label_onehot = tf.transpose(tf.stack([tf.reshape(label[:, 0, ...], [-1]), tf.reshape(label[:, 1, ...], [-1]), tf.reshape(label[:, 2, ...], [-1])]))
    pred = tf.transpose(tf.stack([tf.reshape(pred[:, 0, ...], [-1]), tf.reshape(pred[:, 1, ...], [-1]), tf.reshape(pred[:, 2, ...], [-1])]))

    if class_weights!=None:
        class_weights = tf.convert_to_tensor(class_weights, dtype=label.dtype)
        sample_weights = tf.reduce_sum(tf.multiply(label_onehot, class_weights), axis=1)
        return tf.losses.softmax_cross_entropy(onehot_labels=label_onehot, logits=pred, weights=sample_weights)
    else:
        return tf.losses.softmax_cross_entropy(onehot_labels=label_onehot, logits=pred)
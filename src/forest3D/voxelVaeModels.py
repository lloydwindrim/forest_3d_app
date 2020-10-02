import tensorflow as tf
if tf.__version__.split('.')[0]=='2':
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
    tfd = tf.distributions
else:
    tfd = tf.contrib.distributions

from forest3D import network_ops
from os.path import join

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


class Voxnet_encoder():

    def __init__(self, ph_input, ph_keep_prob, ph_is_training, mdl_name, latent_dims=2, reuse=None, channel_order='channels_last'):

        mdl_name = mdl_name + '_encoder'
        self._activation = layer_activation('lrelu')
        self._useBN = False

        with tf.variable_scope(mdl_name, reuse=reuse):
            ph_input = tf.transpose(ph_input,perm=[0,2,3,4,1]) # swap channels to last dimensions

            self._h1 = network_ops.build_conv_layer(ph_input, kernel_size=5, filters=10, strides=(2, 2, 2),padding='same',
                                                    activation=self._activation, ph_keep_prob=ph_keep_prob, useBN=self._useBN,
                                                    ph_is_training=ph_is_training, dims=3, data_format=channel_order)

            self._h2 = network_ops.build_conv_layer(self._h1, kernel_size=3, filters=10, strides=(1, 1, 1),padding='same',
                                                    activation=self._activation, ph_keep_prob=ph_keep_prob, useBN=self._useBN,
                                                    ph_is_training=ph_is_training, dims=3, data_format=channel_order)

            self._h3 = tf.layers.flatten(self._h2)

            self._loc = tf.layers.dense(self._h3, units=latent_dims, activation=None)
            self._scale = tf.layers.dense(self._h3, units=latent_dims, activation=tf.nn.softplus)
            self._posterior = tfd.MultivariateNormalDiag(self._loc, self._scale)

        self._vars = [var for var in tf.trainable_variables() if var.name.startswith(mdl_name)] + \
                     [var for var in tf.global_variables() if
                      (var.name.startswith(join(mdl_name, 'BatchNorm')) & (var.name.find('moving') != -1))]


    def get_posterior(self):

        return self._posterior

    def get_code(self, type='mean'):

        if type == 'mean':
            return self._posterior.mean()
        elif type == 'sample':
            return self._posterior.sample()

    def get_variance(self):

        return self._scale

    def get_vars(self):

        return self._vars


class Voxnet_decoder():

    def __init__(self, code, output_dims, ph_keep_prob, ph_is_training, mdl_name, reuse=None, channel_order='channels_last'):

        mdl_name = mdl_name + '_decoder'
        self._activation = layer_activation('lrelu')
        self._useBN = False

        d1 = int(output_dims[0] / 2)
        d2 = int(output_dims[2] / 2)

        with tf.variable_scope(mdl_name, reuse=reuse):
            self._h1 = network_ops.build_dense_layer(code, units=d1 * d1 * d2, activation=self._activation,ph_keep_prob=ph_keep_prob,
                                                     useBN=self._useBN, ph_is_training=ph_is_training)

            self._h2 = tf.reshape(self._h1, shape=[-1, d1, d1, d2, 1])

            self._h3 = network_ops.build_deconv_layer(self._h2, kernel_size=3, filters=10, strides=(1, 1, 1),
                                                      activation=self._activation, ph_keep_prob=ph_keep_prob, useBN=self._useBN,
                                                      ph_is_training=ph_is_training, padding='same', dims=3,
                                                      data_format=channel_order)

            self._h4 = network_ops.build_deconv_layer(self._h3, kernel_size=5, filters=10, strides=(2, 2, 2),
                                                      activation=self._activation, ph_keep_prob=ph_keep_prob, useBN=self._useBN,
                                                      ph_is_training=ph_is_training, padding='same', dims=3,
                                                      data_format=channel_order)

            self._output = tf.layers.conv3d(self._h4, kernel_size=5, filters=1, strides=(1, 1, 1),padding='same',
                                            activation=None, use_bias=False,data_format=channel_order)

            self._output = tf.transpose(self._output,perm=[0, 4, 1, 2, 3])  # swap channels to first dimension (after batch)

            self._distribution = tfd.Independent(tfd.Bernoulli(self._output), 3)

        self._vars = [var for var in tf.trainable_variables() if var.name.startswith(mdl_name)] + \
                     [var for var in tf.global_variables() if
                      (var.name.startswith(join(mdl_name, 'BatchNorm')) & (var.name.find('moving') != -1))]


    def get_likelihood(self, data):

        return self._distribution.log_prob(data)


    def get_output(self, type='mean'):
        if type == 'mean':
            return self._distribution.mean()
        elif type == 'sample':
            return self._distribution.sample()


    def get_vars(self):

        return self._vars

import tensorflow as tf
if tf.__version__.split('.')[0]=='2':
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()

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


class Voxnet():

    def __init__(self, ph_input, ph_keep_prob, mdl_name, nClasses=3, reuse=None, channel_order='channels_last'):

        mdl_name = mdl_name + '_model'
        self._activation = layer_activation('lrelu')

        with tf.variable_scope(mdl_name, reuse=reuse):
            ph_input = tf.transpose(ph_input,perm=[0,2,3,4,1]) # swap channels to last dimensions

            self._h1 = tf.layers.conv3d(ph_input, kernel_size=5, filters=10, strides=(2,2,2), padding='same',
                                       activation=self._activation, use_bias=False, data_format=channel_order)

            self._h1_drop = tf.nn.dropout(self._h1, ph_keep_prob)

            self._h2 = tf.layers.conv3d(self._h1_drop, kernel_size=3, filters=10, strides=(1, 1, 1), padding='same',
                                       activation=self._activation, use_bias=False, data_format=channel_order)

            self._h2_drop = tf.nn.dropout(self._h2, ph_keep_prob)

            self._h3 = tf.layers.conv3d_transpose(self._h2_drop, kernel_size=3, filters=10, strides=(1, 1, 1), padding='same',
                                           activation=self._activation, use_bias=False, data_format=channel_order)

            self._h3_drop = tf.nn.dropout(self._h3, ph_keep_prob)

            self._h4 = tf.layers.conv3d_transpose(self._h3_drop, kernel_size=3, filters=10, strides=(2,2,2), padding='same',
                                           activation=self._activation, use_bias=False, data_format=channel_order)

            self._output = tf.layers.conv3d(self._h4, kernel_size=5, filters=nClasses, strides=(1,1,1), padding='same',
                                       activation=None, use_bias=False, data_format=channel_order)

            self._output = tf.transpose(self._output,perm=[0,4,1,2,3]) # swap channels to first dimension (after batch)

        self._vars = [var for var in tf.trainable_variables() if var.name.startswith(mdl_name)]
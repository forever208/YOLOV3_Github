import tensorflow as tf

class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    default as training=False for inference state
    training=True is training state
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


"""conv + batch_normalisation + Leaky_relu"""
def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True, conv_trainable=True):
    """
    :param input_layer: 4D tensor [batch, h, w, c]
    :param filters_shape: (f_size, f_size, input_channels, num_filters)
    :param downsample: True/False
    :param activate: True/False
    :param bn: True/False
    :return: 4D tensor [batch, h, w, c]
    """
    if downsample:    # stride=(2,2), no padding,
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)    # add 0 on top and left
        padding = 'valid'
        strides = 2
    else:    # stride=(1,1), add padding, output original size
        strides = 1
        padding = 'same'

    conv = tf.keras.layers.Conv2D(trainable=conv_trainable,
                                  filters=filters_shape[-1],    # filter number
                                  kernel_size = filters_shape[0],    # filter size
                                  strides=strides, padding=padding,
                                  use_bias=not bn,    # default as no bias
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),    # random normal distribution
                                  bias_initializer=tf.constant_initializer(0.))(input_layer)    # input 4D tensor [batch, h, w, channel]

    if bn: conv = BatchNormalization()(conv)
    if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv    # 4D tensor [batch, height, width, channels]


"""conv + batch_normalisation + Leaky_relu + redidual"""
def residual_block(input_layer, input_channel, filter_num1, filter_num2, conv_trainable=True):
    """
    :return: 4D tensor [bacth, h, w, c]
    """
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1), conv_trainable=conv_trainable)
    conv = convolutional(conv       , filters_shape=(3, 3, filter_num1,   filter_num2), conv_trainable=conv_trainable)

    residual_output = short_cut + conv    # residual block: h(x)=f(x)+x, as f(x)=0 is easier to learn than f(x)=x
    return residual_output


def upsample(input_layer):
    # height and width are doubled by nearest neighbour interpolation
    return tf.image.resize(input_layer,
                           (input_layer.shape[1] * 2, input_layer.shape[2] * 2),    # height*2, width*2
                           method='nearest')
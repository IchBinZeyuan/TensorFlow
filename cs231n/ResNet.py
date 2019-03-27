import tensorflow as tf

def residual_block(x, filters, name = None):
    with tf.variable_scope(name):
        initializer = tf.variance_scaling_initializer(scale=2.0)
        L = tf.layers.batch_normalization(x)
        L = tf.nn.relu(L)
        L = tf.layers.conv2d(L,filters,3, strides = 1, padding = 'same', kernel_initializer = initializer, name = 'conv1')
        L = tf.layers.conv2d(L, filters, 3, strides = 1, padding = 'same',kernel_initializer = initializer, name = 'conv2')
        L += x
        return L

def residual_blocks(x, n, filters = 16, name = None):
    for i in range(n):
        L = residual_block(x, filters = filters, name = name + '{0}'.format(i+1))
        x = L
    return L

def resnet_110(x, keep_prob = None, phase_train = None):
    initializer = tf.variance_scaling_initializer(scale=2.0)
    with tf.variable_scope("resnet_110"):
        conv1 = tf.layers.conv2d(x, 16, 3, 1, kernel_initializer = initializer, name ='conv1')
        with tf.variable_scope('res32_32'):
            conv2 = residual_blocks(conv1, 18, name = 'residual_')
        with tf.variable_scope('res16_16'):
            conv3 = tf.layers.batch_normalization(conv2, name = 'bn_19')
            conv3 = tf.nn.relu(conv3)
            conv3 = tf.layers.conv2d(conv3, 32, 1, 2, kernel_initializer = initializer, name = 'subsample_1')
            conv3 = residual_blocks(conv3, 18, filters = 32, name = 'residual_')

        with tf.variable_scope('res8_8'):
            conv4 = tf.layers.batch_normalization(conv3, name = 'bn_36')
            conv4 = tf.nn.relu(conv3)
            conv4 = tf.layers.conv2d(conv4, 64, 1, 2, kernel_initializer = initializer, name = 'subsample_2')
            conv4 = residual_blocks(conv4, 18, filters = 64, name = 'residual_')
            conv = tf.layers.batch_normalization(conv4, name = 'bn_53')
            conv = tf.nn.relu(conv)

        conv = tf.reduce_mean(conv, [1,2], keep_dims= True)

        output = tf.layers.dense(conv, 10)

    return output

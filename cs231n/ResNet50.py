import tensorflow as tf



def block(x, n_out, n, init_stride = 2, is_training = True):
    h_out = n_out // 4
    out = bottleneck(x, h_out, n_out, stride = init_stride, is_training = is_training)
    for i in range(1,n):
        out = bottleneck(out, h_out, n_out, is_training = is_training)

    return out

def bottleneck(x, h_out, n_out, stride = None, is_training = True):
    n_in = x.get_shape()[-1]
    print("x shape:",x.get_shape())
    if stride is None:
        stride = 1 if n_in == n_out else 2

    h = tf.layers.conv2d(x, h_out, 1, strides = stride, padding = 'same')
    h = tf.layers.batch_normalization(h, training = is_training)
    h = tf.nn.relu(h)

    h = tf.layers.conv2d(h, h_out, 3, strides = 1, padding = 'same')
    h = tf.layers.batch_normalization(h, training=is_training)
    h = tf.nn.relu(h)

    h = tf.layers.conv2d(h, n_out, 1, strides = 1, padding = 'same')
    h = tf.layers.batch_normalization(h, training=is_training)

    if n_in != n_out:
        shortcut = tf.layers.conv2d(x, n_out, 1, strides = stride, padding = 'same')
        shortcut = tf.layers.batch_normalization(shortcut, training = is_training)
    else:
        shortcut = x
    print("shortcut shape:",shortcut.get_shape())
    print("h shape:", h.get_shape())
    return tf.nn.relu(shortcut + h)

def ResNet50(inputs, is_training):
    net = tf.layers.conv2d(inputs, 64, 3, 1, padding = 'same')
    net = tf.nn.relu(net)
    net = tf.layers.max_pooling2d(net, 2, 2)
    print("net shape:",net.get_shape())
    net = block(net, 256, 3, init_stride = 1, is_training = is_training)
    net = block(net, 512, 4, is_training = is_training)
    net = block(net, 1024, 6, is_training = is_training)
    net = block(net, 2048, 3, is_training = is_training)
    net = tf.layers.average_pooling2d(net, 2, 2)
    net = tf.squeeze(net, [1,2])
    logits = tf.layers.dense(net, 10)
    return logits
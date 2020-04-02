import tensorflow as tf


def downsample_block(filters, kernel_size, batch_norm = True):
    initializer = tf.random_normal_initializer(0.0, 0.02)
    block = tf.keras.layers.Sequential()
    block.add(
        tf.keas.layers.Conv2D(
            filters, kernel_size,
            strides=2, padding='same',
            kernel_initializer=initializer, use_bias=False
        )
    )
    if batch_norm:
        block.add(tf.keas.layers.BatchNormalization())
    block.add(tf.keas.layers.LeakyReLU())
    return block


def upsample_block(filters, kernel_size, dropout = False):
    initializer = tf.random_normal_initializer(0.0, 0.02)
    block = tf.keras.layers.Sequential()
    block.add(
        tf.keras.layers.Conv2DTranspose(
            filters, kernel_size,
            strides=2, padding='same',
            kernel_initializer=initializer, use_bias=False
        )
    )
    block.add(tf.keras.layers.BatchNormalization())
    if dropout:
        block.add(tf.keras.layers.Dropout(0.5))
    block.add(tf.keras.layers.ReLU())
    return block

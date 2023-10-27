#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout

def dropout_create_layer(prev, n, activation, keep_prob):
    reg = Dropout(rate=keep_prob)
    init = tf.keras.initializers.VarianceScaling(mode="fan_avg")

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init
    )(prev)
    output = reg(layer)

    return output

if __name__ == '__main__':
    tf.random.set_seed(0)
    np.random.seed(0)
    x = tf.placeholder(tf.float32, shape=[None, 784])
    X = np.random.randint(0, 256, size=(10, 784))
    a = dropout_create_layer(x, 256, tf.nn.tanh, 0.8)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(a, feed_dict={x: X}))

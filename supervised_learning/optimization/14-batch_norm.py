#!/usr/bin/env python3
"""
Defines a function create_batch_norm_layer
"""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=init)
    z = layer(prev)

    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True)
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True)

    mean, variance = tf.nn.moments(z, axes=[0])
    epsilon = 1e-8
    z_norm = tf.nn.batch_normalization(z, mean, variance, beta, gamma, epsilon)

    return activation(z_norm)

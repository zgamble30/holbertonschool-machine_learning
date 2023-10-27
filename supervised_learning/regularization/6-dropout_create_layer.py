#!/usr/bin/env python3
"""create neural net using dropout"""
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Create a layer of a neural network using dropout.

    Args:
        prev: A tensor containing the output of the previous layer.
        n: The number of nodes the new layer should contain.
        activation: The activation function to be used on the layer.
        keep_prob: The probability that a node will be kept.

    Returns:
        The output of the new layer.
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation, kernel_initializer=init)(prev)
    dropout = tf.layers.Dropout(rate=1 - keep_prob)(layer)
    return dropout

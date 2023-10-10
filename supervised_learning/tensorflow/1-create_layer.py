#!/usr/bin/env python3
"""Module containing a function to create a layer in a neural network."""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    Creates a layer in a neural network.

    Args:
        prev (tf.Tensor): The tensor output of the previous layer.
        n (int): The number of nodes in the layer to create.
        activation: The activation function that the layer should use.

    Returns:
        tf.Tensor: The tensor output of the layer.
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init,
        name='layer'
    )
    output = layer(prev)
    return output

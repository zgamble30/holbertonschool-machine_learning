#!/usr/bin/env python3
"""Module containing a function for forward propagation in a neural network."""
import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network.

    Args:
        x (tf.Tensor): Placeholder for the input data.
        layer_sizes (list): List containing the number of nodes in each layer.
        activations (list): List containing the activation functions.

    Returns:
        tf.Tensor: The prediction of the network in tensor form.
    """
    prev = x
    for i in range(len(layer_sizes)):
        prev = create_layer(prev, layer_sizes[i], activations[i])

    return prev

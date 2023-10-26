#!/usr/bin/env python3
"""Create a Layer with L2 Regularization"""

import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Create a TensorFlow layer that includes L2 regularization:
    Args:
        prev: A tensor containing the output of the previous layer.
        n: The number of nodes the new layer should contain.
        activation: The activation function that should be used on the layer.
        lambtha: The L2 regularization parameter.
    Returns:
        The output of the new layer.
    """

    # Define an L2 regularization term
    l2_regularizer = tf.keras.regularizers.l2(lambtha)

    # Initialize weights using Variance Scaling
    weights_init = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg")

    # Create a dense layer with L2 regularization
    l2_regularized_layer = tf.layers.Dense(
        n,
        activation=activation,
        kernel_initializer=weights_init,
        kernel_regularizer=l2_regularizer,
        name="custom_layer"
    )

    # Apply the layer to the previous layer
    return l2_regularized_layer(prev)

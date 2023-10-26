#!/usr/bin/env python3
"""Create a Layer with L2 Regularization"""

import tensorflow.compat.v1 as tf


def custom_l2_regularized_layer(previous_layer, num_units, activation, regularization_lambda):
    """
    Create a TensorFlow layer that includes L2 regularization:
    Args:
        previous_layer: The previous layer to connect to.
        num_units: The number of units (neurons) in the layer.
        activation: The activation function to use for the layer.
        regularization_lambda: The L2 regularization parameter.
    Returns:
        A TensorFlow layer with L2 regularization applied.
    """
    
    # Define an L2 regularization term
    l2_regularizer = tf.keras.regularizers.l2(regularization_lambda)
    
    # Initialize weights using Variance Scaling
    weights_init = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg")
    
    # Create a dense layer with L2 regularization
    l2_regularized_layer = tf.layers.Dense(
        num_units,
        activation=activation,
        kernel_initializer=weights_init,
        kernel_regularizer=l2_regularizer,
        name="custom_layer"
    )
    
    # Apply the layer to the previous layer
    return l2_regularized_layer(previous_layer)

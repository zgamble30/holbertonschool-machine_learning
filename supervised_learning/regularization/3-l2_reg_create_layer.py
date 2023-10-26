#!/usr/bin/env python3
"""Create a Layer with L2 Regularization"""

import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """creates a TensorFlow layer that includes L2 regularization:"""
    
    # Define an L2 regularization term
    l2_reg = tf.keras.regularizers.l2(lambtha)
    
    # Initialize weights using Variance Scaling
    weights_initializer = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg")
    
    # Create a dense layer with L2 regularization
    layer = tf.layers.Dense(
        n,
        activation=activation,
        kernel_initializer=weights_initializer,
        kernel_regularizer=l2_reg,
        name="layer"
    )
    
    # Apply the layer to the previous layer
    return layer(prev)

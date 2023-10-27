#!/usr/bin/env python3
"""create neural net using dropout"""
import tensorflow.compat.v1 as tf


def create_dropout_layer(previous_layer, units, activation, keep_probability):
    """
    Creates a layer using dropout.

    Args:
        previous_layer: The previous layer in the network.
        units: The number of nodes in the new layer.
        activation: The activation function for the layer.
        keep_probability: The probability that a node will be kept (dropout rate).

    Returns:
        The output layer with dropout regularization.
    """
    dropout_rate = 1 - keep_probability
    dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)
    initializer = tf.keras.initializers.VarianceScaling(mode="fan_avg")

    layer = tf.keras.layers.Dense(
        units=units,
        activation=activation,
        kernel_regularizer=dropout_layer,
        kernel_initializer=initializer
    )(previous_layer)

    return layer

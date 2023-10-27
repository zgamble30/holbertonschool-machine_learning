#!/usr/bin/env python3
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """creates layer using dropout"""
    dropout_rate = 1 - keep_prob
    dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)
    initializer = tf.keras.initializers.VarianceScaling(mode="fan_avg")

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=dropout_layer,
        kernel_initializer=initializer
    )(prev)

    return layer

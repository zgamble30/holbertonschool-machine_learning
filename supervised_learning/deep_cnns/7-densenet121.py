#!/usr/bin/env python3
"""
Builds a transition layer as described in Densely Connected Convolutional Networks
"""
import tensorflow.keras as K


def transition_layer(prev_output, num_filters, compression_factor):
    """
    Builds a transition layer as described in Densely Connected Convolutional Networks

    Args:
        prev_output (tensor): Output from the previous layer
        num_filters (int): Number of filters in the previous layer
        compression_factor (float): Compression factor for the transition layer

    Returns:
        tensor: The output of the transition layer
        int: The number of filters within the output
    """
    # He normal initializer
    he_normal = K.initializers.he_normal()

    # Batch Normalization
    x = K.layers.BatchNormalization(axis=3)(prev_output)

    # ReLU activation
    x = K.layers.Activation('relu')(x)

    # 1x1 Convolution with compression
    num_filters = int(num_filters * compression_factor)
    x = K.layers.Conv2D(
        filters=num_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer=he_normal
    )(x)

    # Average pooling
    x = K.layers.AveragePooling2D(
        pool_size=2,
        strides=2,
        padding='valid'
    )(x)

    return x, num_filters

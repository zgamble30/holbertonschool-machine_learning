#!/usr/bin/env python3
"""
Defines the projection_block function
"""

import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in Deep Residual Learning for
    Image Recognition (2015)

    Arguments:
        A_prev: Output from the previous layer
        filters: Tuple or list (F11, F3, F12)
                 F11 is the number of filters in the first 1x1 convolution
                 F3 is the number of filters in the 3x3 convolution
                 F12 is the number of filters in the second 1x1 convolution
        s: Stride of the first convolution in both the main path and
           shortcut connection

    Returns:
        Activated output of the projection block
    """
    # Retrieve filters
    F11, F3, F12 = filters

    # Shortcut path
    shortcut = K.layers.Conv2D(
        F12,
        kernel_size=(1, 1),
        strides=(s, s),
        kernel_initializer='he_normal'
    )(A_prev)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    # Main path
    conv1 = K.layers.Conv2D(
        F11,
        kernel_size=(1, 1),
        strides=(s, s),
        kernel_initializer='he_normal'
    )(A_prev)
    conv1 = K.layers.BatchNormalization(axis=3)(conv1)
    conv1 = K.layers.Activation('relu')(conv1)

    conv2 = K.layers.Conv2D(
        F3,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer='he_normal'
    )(conv1)
    conv2 = K.layers.BatchNormalization(axis=3)(conv2)
    conv2 = K.layers.Activation('relu')(conv2)

    conv3 = K.layers.Conv2D(
        F12,
        kernel_size=(1, 1),
        kernel_initializer='he_normal'
    )(conv2)
    conv3 = K.layers.BatchNormalization(axis=3)(conv3)

    # Add shortcut to main path
    output = K.layers.Add()([conv3, shortcut])
    output = K.layers.Activation('relu')(output)

    return output

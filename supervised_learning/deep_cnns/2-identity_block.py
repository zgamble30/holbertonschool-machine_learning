#!/usr/bin/env python3
"""Builds an identity block."""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """Build an identity block.

    Args:
        A_prev (Keras tensor): Output from the previous layer.
        filters (list/tuple): F11, F3, F12, respectively:
            F11: Number of filters in the first 1x1 convolution.
            F3: Number of filters in the 3x3 convolution.
            F12: Number of filters in the second 1x1 convolution.

    Returns:
        Keras tensor: Activated output of the identity block.
    """
    F11, F3, F12 = filters

    init = K.initializers.he_normal(seed=None)

    conv1x1 = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=init
    )(A_prev)

    batch_norm1 = K.layers.BatchNormalization(axis=3)(conv1x1)
    activation1 = K.layers.Activation('relu')(batch_norm1)

    conv3x3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer=init
    )(activation1)

    batch_norm2 = K.layers.BatchNormalization(axis=3)(conv3x3)
    activation2 = K.layers.Activation('relu')(batch_norm2)

    conv1x1_2 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=init
    )(activation2)

    batch_norm3 = K.layers.BatchNormalization(axis=3)(conv1x1_2)

    add = K.layers.Add()([batch_norm3, A_prev])

    activation3 = K.layers.Activation('relu')(add)

    return activation3

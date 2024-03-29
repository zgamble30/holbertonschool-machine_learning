#!/usr/bin/env python3
"""
Builds a projection block as described in Deep
Residual Learning for Image
Recognition (2015)
"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    Args:
        A_prev: output from the previous layer.
        filters: tuple or list containing F11, F3,
        F12, respectively.
        s: stride of the first convolution
        in both the main path and the shortcut
        connection.

    Returns:
        the activated output of the projection block.
    """
    F11, F3, F12 = filters

    # Initialize the He normal initializer
    initializer = K.initializers.he_normal()

    # First component of main path
    conv2d = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='valid',
        kernel_initializer=initializer
    )(A_prev)
    batch_normalization = K.layers.BatchNormalization(axis=3)(conv2d)
    activation = K.layers.Activation('relu')(batch_normalization)

    # Second component of main path
    conv2d_1 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(activation)
    batch_normalization_1 = K.layers.BatchNormalization(axis=3)(conv2d_1)
    activation_1 = K.layers.Activation('relu')(batch_normalization_1)

    # Third component of main path
    conv2d_2 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        kernel_initializer=initializer
    )(activation_1)
    batch_normalization_2 = K.layers.BatchNormalization(axis=3)(conv2d_2)

    # Shortcut path
    conv2d_3 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='valid',
        kernel_initializer=initializer
    )(A_prev)
    batch_normalization_3 = K.layers.BatchNormalization(axis=3)(conv2d_3)

    # Final step: Add shortcut value to main path
    add = K.layers.Add()([batch_normalization_2, batch_normalization_3])
    activation_2 = K.layers.Activation('relu')(add)

    return activation_2

#!/usr/bin/env python3
"""
Builds a transition layer as described in Densely Connected Convolutional Networks
"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in Densely Connected Convolutional Networks

    Arguments:
    - X: output from the previous layer
    - nb_filters: integer representing the number of filters in X
    - compression: compression factor for the transition layer

    Returns:
    - The output of the transition layer
    - The number of filters within the output, respectively
    """
    HeNormal = K.initializers.he_normal()

    # Batch Normalization
    X = K.layers.BatchNormalization(axis=3)(X)

    # ReLU activation
    X = K.layers.Activation('relu')(X)

    # Convolution with compression
    nb_filters = int(nb_filters * compression)
    X = K.layers.Conv2D(
        filters=nb_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer=HeNormal
    )(X)

    # Average pooling
    X = K.layers.AveragePooling2D(
        pool_size=2,
        strides=2,
        padding='valid'
    )(X)

    return X, nb_filters

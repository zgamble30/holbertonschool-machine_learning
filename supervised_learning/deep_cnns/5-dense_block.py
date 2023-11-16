#!/usr/bin/env python3
"""
Builds a dense block as described in Densely Connected Convolutional Networks
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in Densely Connected Convolutional Networks

    Arguments:
    - X: output from the previous layer
    - nb_filters: integer representing the number of filters in X
    - growth_rate: growth rate for the dense block
    - layers: number of layers in the dense block

    Returns:
    - The concatenated output of each layer within the Dense Block
    - The number of filters within the concatenated outputs, respectively
    """
    HeNormal = K.initializers.he_normal()

    # List to store the concatenated outputs
    concat_outputs = [X]

    for _ in range(layers):
        # Batch Normalization
        X = K.layers.BatchNormalization(axis=3)(X)
        # ReLU activation
        X = K.layers.Activation('relu')(X)
        # Convolution with bottleneck layer
        bottleneck = K.layers.Conv2D(
            filters=growth_rate * 4,
            kernel_size=1,
            strides=1,
            padding='same',
            kernel_initializer=HeNormal
        )(X)
        # Batch Normalization
        bottleneck = K.layers.BatchNormalization(axis=3)(bottleneck)
        # ReLU activation
        bottleneck = K.layers.Activation('relu')(bottleneck)
        # Convolution
        conv = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=3,
            strides=1,
            padding='same',
            kernel_initializer=HeNormal
        )(bottleneck)
        # Concatenate the output of the current layer to the list
        concat_outputs.append(conv)
        # Update the number of filters
        nb_filters += growth_rate

    # Concatenate all the outputs along the last axis
    concat_block = K.layers.concatenate(concat_outputs, axis=3)

    return concat_block, nb_filters

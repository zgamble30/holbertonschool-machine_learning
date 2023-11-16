#!/usr/bin/env python3
"""
buiilds dense block
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in
    Densely Connected Convolutional Networks.

    Args:
        X (tf.Tensor): The output from the previous layer.
        nb_filters (int): Number of filters in X.
        growth_rate (int): Growth rate for the dense block.
        layers (int): Number of layers in the dense block.

    Returns:
        A tuple containing the concatenated
        output of each layer within the Dense Block
        and the number of filters within the
        concatenated outputs, respectively.
    """
    # Function for He normal initialization
    HeNormal = K.initializers.he_normal()

    # List to store the concatenated outputs
    concat_outputs = [X]

    # Loop through the specified number of layers in the dense block
    for _ in range(layers):
        # Batch Normalization
        X = K.layers.BatchNormalization(axis=3)(X)
        X = K.layers.Activation('relu')(X)

        # 1x1 bottleneck convolution
        X = K.layers.Conv2D(
            filters=4 * growth_rate,
            kernel_size=1,
            strides=1,
            padding="same",
            kernel_initializer=HeNormal,
        )(X)

        # Batch Normalization
        X = K.layers.BatchNormalization(axis=3)(X)
        X = K.layers.Activation('relu')(X)

        # 3x3 convolution
        X = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=HeNormal,
        )(X)

        # Append the output to the list
        concat_outputs.append(X)

        # Concatenate the outputs along the last axis
        X = K.layers.Concatenate(axis=3)(concat_outputs)

        # Update the number of filters
        nb_filters += growth_rate

    return X, nb_filters

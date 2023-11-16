#!/usr/bin/env python3
"""
Builds a dense block as described in Densely Connected Convolutional Networks
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Args:
        X: output from the previous layer.
        nb_filters: integer representing the number of filters in X.
        growth_rate: growth rate for the dense block.
        layers: number of layers in the dense block.
    
    Returns:
        The concatenated output of each layer within the Dense Block and 
        the number of filters within the concatenated outputs, respectively.
    """
    init = K.initializers.he_normal(seed=None)

    for i in range(layers):

        norm1 = K.layers.BatchNormalization()(X)
        act1 = K.layers.Activation('relu')(norm1)
        conv1 = K.layers.Conv2D(filters=4 * growth_rate,
                                kernel_size=1,
                                padding='same',
                                kernel_initializer=init)(act1)

        norm2 = K.layers.BatchNormalization()(conv1)
        act2 = K.layers.Activation('relu')(norm2)
        conv2 = K.layers.Conv2D(filters=growth_rate,
                                kernel_size=3,
                                padding='same',
                                kernel_initializer=init)(act2)

        X = K.layers.concatenate([X, conv2])
        nb_filters += growth_rate

    return X, nb_filters

#!/usr/bin/env python3
"""
Builds the DenseNet-121 architecture as described
in Densely Connected Convolutional Networks
"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture as described in
    Densely Connected Convolutional Networks

    Args:
        growth_rate (int): Growth rate for dense blocks
        compression (float): Compression factor for transition layers

    Returns:
        keras.Model: The DenseNet-121 model
    """
    # He normal initializer
    HeNormal = K.initializers.he_normal()

    # Input layer
    X_input = K.layers.Input((224, 224, 3))

    # Initial Convolution
    X = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        strides=2,
        padding='same',
        kernel_initializer=HeNormal
    )(X_input)

    # Batch Normalization and ReLU
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # MaxPooling
    X = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(X)

    # Dense Block 1
    X, nb_filters = dense_block(X, 64, growth_rate, 6)

    # Transition Layer 1
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 2
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)

    # Transition Layer 2
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 3
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)

    # Transition Layer 3
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 4
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    # Global Average Pooling
    X = K.layers.GlobalAveragePooling2D()(X)

    # Fully connected layer
    X = K.layers.Dense(units=1000, activation='softmax', kernel_initializer=HeNormal)(X)

    # Model
    model = K.models.Model(inputs=X_input, outputs=X)

    return model

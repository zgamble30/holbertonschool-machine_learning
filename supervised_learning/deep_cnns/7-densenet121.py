#!/usr/bin/env python3
"""
Builds densenet-121 model
"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture.

    Args:
        growth_rate: Growth rate for the dense blocks.
        compression: Compression factor for the transition layers.

    Returns:
        Keras Model: The DenseNet-121 model.
    """
    HeNormal = K.initializers.he_normal()

    # Input layer
    input_layer = K.layers.Input(shape=(224, 224, 3))

    # Initial convolution layer
    X = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        strides=2,
        padding="same",
        kernel_initializer=HeNormal,
    )(input_layer)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding="same",
    )(X)

    # First dense block
    X, nb_filters = dense_block(X, 64, growth_rate, 6)

    # First transition layer
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Second dense block
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)

    # Second transition layer
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Third dense block
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)

    # Third transition layer
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Fourth dense block
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    # Global average pooling
    X = K.layers.GlobalAveragePooling2D()(X)

    # Fully connected layer
    X = K.layers.Dense(units=1000, activation='softmax',
                       kernel_initializer=HeNormal)(X)

    # Create and return the model
    model = K.models.Model(inputs=input_layer, outputs=X, name='DenseNet121')
    return model

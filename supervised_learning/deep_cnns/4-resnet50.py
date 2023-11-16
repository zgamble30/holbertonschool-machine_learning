#!/usr/bin/env python3
"""resnet-50 architecture"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described
    in Deep Residual Learning for Image Recognition (2015).

    Returns:
        A Keras model representing the ResNet-50 architecture.
    """
    HeNormal = K.initializers.he_normal()
    X_input = K.layers.Input((224, 224, 3))

    # Initial Convolutional Layer
    X = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        strides=2,
        padding="same",
        kernel_initializer=HeNormal,
    )(X_input)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding="same",
    )(X)

    # First Block
    X = projection_block(X, [64, 64, 256], s=1)
    X = identity_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])

    # Second Block
    X = projection_block(X, [128, 128, 512], s=2)
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])

    # Third Block
    X = projection_block(X, [256, 256, 1024], s=2)
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])

    # Fourth Block
    X = projection_block(X, [512, 512, 2048], s=2)
    X = identity_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])

    # Global Average Pooling and Dense Layer
    X = K.layers.AveragePooling2D(
        pool_size=7,
        strides=1,
        padding="valid",
    )(X)
    X = K.layers.Dense(units=1000, activation='softmax',
                       kernel_initializer=HeNormal)(X)

    model = K.models.Model(inputs=X_input, outputs=X)

    return model

#!/usr/bin/env python3
"""
Builds the ResNet-50 architecture as described
in Deep Residual Learning for Image Recognition (2015)
"""
import tensorflow.keras as K
# Importing the identity and projection block functions
identity_block_func = __import__('2-identity_block').identity_block
projection_block_func = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture.

    Returns:
        Keras Model: The ResNet-50 model.
    """
    # He normal initialization function for the weights
    he_normal_initializer = K.initializers.he_normal()

    # Input layer for 224x224 images with 3 channels
    input_layer = K.layers.Input((224, 224, 3))

    # Initial convolution layer with 64 filters, and same padding
    X = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        strides=2,
        padding="same",
        kernel_initializer=he_normal_initializer,
    )(input_layer)

    # Batch normalization and ReLU activation
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Max pooling layer with 3x3 pool size, stride 2, and same padding
    X = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding="same",
    )(X)

    # First set of residual blocks (Conv2)
    X = projection_block_func(X, [64, 64, 256], s=1)
    X = identity_block_func(X, [64, 64, 256])
    X = identity_block_func(X, [64, 64, 256])

    # Second set of residual blocks (Conv3)
    X = projection_block_func(X, [128, 128, 512], s=2)
    X = identity_block_func(X, [128, 128, 512])
    X = identity_block_func(X, [128, 128, 512])
    X = identity_block_func(X, [128, 128, 512])

    # Third set of residual blocks (Conv4)
    X = projection_block_func(X, [256, 256, 1024], s=2)
    X = identity_block_func(X, [256, 256, 1024])
    X = identity_block_func(X, [256, 256, 1024])
    X = identity_block_func(X, [256, 256, 1024])
    X = identity_block_func(X, [256, 256, 1024])
    X = identity_block_func(X, [256, 256, 1024])

    # Fourth set of residual blocks (Conv5)
    X = projection_block_func(X, [512, 512, 2048], s=2)
    X = identity_block_func(X, [512, 512, 2048])
    X = identity_block_func(X, [512, 512, 2048])

    # Average pooling layer with 7x7 pool size, stride 1, and valid padding
    X = K.layers.AveragePooling2D(
        pool_size=7,
        strides=1,
        padding="valid",
    )(X)

    # Fully connected layer with 1000 units
    X = K.layers.Dense(units=1000, activation='softmax',
                       kernel_initializer=he_normal_initializer)(X)

    # Creating the Keras Model instance
    model = K.models.Model(inputs=input_layer, outputs=X, name='ResNet50')

    return model

#!/usr/bin/env python3
"""
Builds the ResNet-50 architecture as described
    in Deep Residual Learning for Image Recognition (2015)
"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture.

    Returns:
        Keras Model: The ResNet-50 model.
    """
    # Function for He normal initialization
    he_normal_initializer = K.initializers.he_normal()

    # Input layer
    input_layer = K.layers.Input((224, 224, 3))

    # Initial convolution layer
    X = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        strides=2,
        padding="same",
        kernel_initializer=he_normal_initializer,
    )(input_layer)
    # Batch normalization before activation function
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    # Max pooling
    X = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding="same",
    )(X)

    # Conv2 block
    X = projection_block(X, [64, 64, 256], s=1)
    X = identity_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])

    # Conv3 block
    X = projection_block(X, [128, 128, 512], s=2)
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])

    # Conv4 block
    X = projection_block(X, [256, 256, 1024], s=2)
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])

    # Conv5 block
    X = projection_block(X, [512, 512, 2048], s=2)
    X = identity_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])

    # Average pooling
    X = K.layers.AveragePooling2D(
        pool_size=7,
        strides=1,
        padding="valid",
    )(X)

    # Fully connected layer
    X = K.layers.Dense(units=1000, activation='softmax',
                       kernel_initializer=he_normal_initializer)(X)

    # Create and return the model
    model = K.models.Model(inputs=input_layer, outputs=X, name='ResNet50')
    return model

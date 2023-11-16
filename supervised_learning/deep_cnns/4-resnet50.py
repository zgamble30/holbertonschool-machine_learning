#!/usr/bin/env python3
"""
buiilds resnet-50 architecture
"""
import tensorflow.keras as K
from supervised_learning.deep_cnns.2-identity_block import identity_block
from supervised_learning.deep_cnns.3-projection_block import projection_block


def resnet50():
    """
    Function that builds the ResNet-50 architecture as described
    in Deep Residual Learning for Image Recognition (2015)
    """

    # Define the input
    X_input = K.Input(shape=(224, 224, 3))

    # Zero-Padding
    X = K.layers.ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = K.layers.Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer='he_normal')(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = projection_block(X, [64, 64, 256], s=1)
    X = identity_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])

    # Stage 3
    X = projection_block(X, [128, 128, 512])
    for i in range(3):
        X = identity_block(X, [128, 128, 512])

    # Stage 4
    X = projection_block(X, [256, 256, 1024])
    for i in range(5):
        X = identity_block(X, [256, 256, 1024])

    # Stage 5
    X = projection_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])

    # AVGPOOL
    X = K.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    # output layer
    X = K.layers.Flatten()(X)
    X = K.layers.Dense(1000, activation='softmax', kernel_initializer='he_normal')(X)

    # Create model
    model = K.models.Model(inputs=X_input, outputs=X, name='ResNet50')

    return model

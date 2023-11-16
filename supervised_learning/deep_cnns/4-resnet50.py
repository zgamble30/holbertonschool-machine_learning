#!/usr/bin/env python3
"""
buiilds resnet-50 architecture
"""
import tensorflow.keras as K
from __import__('2-identity_block').identity_block
from __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture.

    Returns:
        Keras Model: The ResNet-50 model.
    """
    # Function for He normal initialization
    HeNormal = K.initializers.he_normal(seed=None)

    # Input layer
    X_input = K.layers.Input(shape=(224, 224, 3))

    # Initial convolution layer
    X = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        strides=2,
        padding="same",
        kernel_initializer=HeNormal,
        kernel_regularizer=K.regularizers.l2(1e-4)
    )(X_input)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(X)

    # ResNet blocks
    filters = [64, 128, 256, 512]
    blocks = [3, 4, 6, 3]

    for i in range(4):
        s = 1 if i == 0 else 2
        X = projection_block(X, filters[i], s=s)
        for _ in range(1, blocks[i]):
            X = identity_block(X, filters[i])

    # Average pooling layer
    X = K.layers.AveragePooling2D(pool_size=7, strides=1, padding="valid")(X)

    # Fully connected layer
    X = K.layers.Dense(units=1000, activation='softmax', kernel_initializer=HeNormal)(X)

    # Create and return the model
    model = K.models.Model(inputs=X_input, outputs=X, name='ResNet50')
    return model
 
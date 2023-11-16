#!/usr/bin/env python3
"""
Defines the resnet50 function
"""

import tensorflow.keras as K
from tensorflow.keras.layers import Input


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in Deep Residual Learning for
    Image Recognition (2015)

    Arguments:
        A_prev: Output from the previous layer
        filters: Tuple or list (F11, F3, F12)
                 F11 is the number of filters in the first 1x1 convolution
                 F3 is the number of filters in the 3x3 convolution
                 F12 is the number of filters in the second 1x1 convolution

    Returns:
        Activated output of the identity block
    """
    # Implementation of identity block

def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in Deep Residual Learning for
    Image Recognition (2015)

    Arguments:
        A_prev: Output from the previous layer
        filters: Tuple or list (F11, F3, F12)
                 F11 is the number of filters in the first 1x1 convolution
                 F3 is the number of filters in the 3x3 convolution
                 F12 is the number of filters in the second 1x1 convolution
        s: Stride of the first convolution in both the main path and
           shortcut connection

    Returns:
        Activated output of the projection block
    """
    # Implementation of projection block

def resnet50():
    """
    Builds the ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition (2015)

    Returns:
        Keras model
    """
    # Input layer
    X_input = Input(shape=(224, 224, 3))

    # Initial convolution and batch normalization
    X = K.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2),
                        padding='same', kernel_initializer='he_normal')(X_input)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)

    # Projection blocks
    X = projection_block(X, [64, 64, 256], s=1)
    X = identity_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])

    X = projection_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])

    X = projection_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])

    X = projection_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])

    # Global average pooling
    X = K.layers.AveragePooling2D((2, 2))(X)

    # Flatten layer
    X = K.layers.Flatten()(X)

    # Fully connected layer
    X = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer='he_normal')(X)

    # Create the model
    model = K.models.Model(inputs=X_input, outputs=X, name='ResNet50')

    return model

# Uncomment the following line if you want to see the model summary
# resnet50().summary()

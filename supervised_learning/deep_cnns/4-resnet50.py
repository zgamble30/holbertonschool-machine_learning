#!/usr/bin/env python3
"""builds the resnet 50 architecture"""

import tensorflow.keras as K


def resnet50():
    # Input layer
    X_input = K.Input(shape=(224, 224, 3))

    # Stage 1
    X = K.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', kernel_initializer='he_normal')(X_input)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)

    # Stage 2
    X = projection_block(X, [64, 64, 256], s=1)
    X = identity_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])

    # Stage 3
    X = projection_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])

    # Stage 4
    X = projection_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])

    # Stage 5
    X = projection_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])

    # Average pooling
    X = K.layers.AveragePooling2D((7, 7))(X)

    # Output layer
    X = K.layers.Flatten()(X)
    X_output = K.layers.Dense(1000, activation='softmax', kernel_initializer='he_normal')(X)

    # Create model
    model = K.models.Model(inputs=X_input, outputs=X_output)

    return model

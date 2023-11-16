#!/usr/bin/env python3
"""resnet-50 architecture"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds a ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition (2015)
    """
    X_input = K.Input(shape=(224, 224, 3))

    # Stage 1
    X = K.layers.Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer='he_normal')(X_input)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = projection_block(X, 64, 1)
    X = identity_block(X, 64, 3)

    # Stage 3
    X = projection_block(X, 128, 2)
    X = identity_block(X, 128, 4)

    # Stage 4
    X = projection_block(X, 256, 2)
    X = identity_block(X, 256, 6)

    # Stage 5
    X = projection_block(X, 512, 2)
    X = identity_block(X, 512, 3)

    # AVGPOOL
    X = K.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    # output layer
    X = K.layers.Flatten()(X)
    X = K.layers.Dense(units=1000, activation='softmax', kernel_initializer='he_normal')(X)

    # Create model
    model = K.models.Model(inputs=X_input, outputs=X, name='ResNet50')

    return model

#!/usr/bin/env python3
"""builds a modified version of the LeNet-5 architecture using keras"""
import tensorflow.keras as K


def lenet5(X):
    """
    Build a modified LeNet-5 architecture using Keras.

    Args:
    - X: K.Input of shape (m, 28, 28, 1) containing the input images for the network

    Returns:
    - model: K.Model compiled to use Adam optimization and accuracy metrics
    """

    initializer = K.initializers.he_normal(seed=None)

    # Convolutional Layer 1
    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=initializer
    )(X)

    # Max Pooling Layer 1
    pool1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv1)

    # Convolutional Layer 2
    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=initializer
    )(pool1)

    # Max Pooling Layer 2
    pool2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv2)

    # Flatten layer
    flatten = K.layers.Flatten()(pool2)

    # Fully Connected Layer 1
    fc1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=initializer
    )(flatten)

    # Fully Connected Layer 2
    fc2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=initializer
    )(fc1)

    # Output Layer
    output_layer = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=initializer
    )(fc2)

    # Create the model
    model = K.Model(inputs=X, outputs=output_layer)

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

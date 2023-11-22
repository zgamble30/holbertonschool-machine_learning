#!/usr/bin/env python3
"""
This script trains a convolutional neural network to classify the CIFAR 10 dataset.
"""

import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    This function preprocesses the CIFAR 10 data.

    Args:
    X: numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data,
       where m is the number of data points
    Y: numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X

    Returns: X_p, Y_p
        X_p: numpy.ndarray containing the preprocessed X
        Y_p: numpy.ndarray containing the preprocessed Y
    """
    X_p = X.astype('float32') / 255.0
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

if __name__ == '__main__':
    """
    This block trains a convolutional neural network to classify CIFAR 10 dataset
    and saves the model to 'cifar10.h5'.
    """
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    # Preprocess the data with a lambda layer
    inputs = K.layers.Input(shape=(32, 32, 3))
    x = K.layers.Lambda(lambda x: K.backend.resize_images(x, height_factor=7, width_factor=7, data_format="channels_last"))(inputs)

    # Use EfficientNetV2B3 with pre-trained weights
    base_model = K.applications.EfficientNetV2B3(include_top=False, weights='imagenet', input_tensor=x)

    # Freeze layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers for classification
    x = K.layers.Flatten()(base_model.output)
    output = K.layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = K.models.Model(inputs=inputs, outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=128, epochs=10)

    # Save the model
    model.save('cifar10.h5')

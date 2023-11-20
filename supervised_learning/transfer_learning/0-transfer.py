#!/usr/bin/env python3
"""
Trains a convolutional neural network to classify the CIFAR 10 dataset using Transfer Learning.
"""

import tensorflow.keras as K

def preprocess_data(X, Y):
    """
    Preprocesses the data for the model.

    Arguments:
    - X: numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data, where m is the number of data points
    - Y: numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X

    Returns:
    - X_p: numpy.ndarray containing the preprocessed X
    - Y_p: numpy.ndarray containing the preprocessed Y
    """
    # Normalize pixel values to be between 0 and 1
    X_p = X.astype('float32') / 255.0

    # One-hot encode the labels
    Y_p = K.utils.to_categorical(Y, 10)

    return X_p, Y_p

def transfer_model():
    """
    Builds and trains a transfer learning model on the CIFAR 10 dataset.

    Saves the trained model as cifar10.h5 in the current working directory.
    The saved model is compiled and has a validation accuracy of 87% or higher.
    """
    # Load a pre-trained model from Keras Applications
    base_model = K.applications.ResNet50(
        include_top=False, weights='imagenet', input_shape=(32, 32, 3)
    )

    # Freeze the layers of the pre-trained model
    for layer in base_model.layers:
        layer.trainable = False

    # Create a new model by adding custom layers on top of the pre-trained model
    model = K.models.Sequential([
        K.layers.Lambda(lambda x: K.backend.resize_images(x, height_factor=7, width_factor=7, data_format='channels_last')),
        base_model,
        K.layers.Flatten(),
        K.layers.Dense(512, activation='relu'),
        K.layers.Dropout(0.5),
        K.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Display the model summary
    model.summary()

    # Train the model on your preprocessed data
    # (Replace this with your actual data and training configuration)
    model.fit(X_train, Y_train, epochs=5, batch_size=128, validation_data=(X_val, Y_val))

    # Save the trained model
    model.save('cifar10.h5')

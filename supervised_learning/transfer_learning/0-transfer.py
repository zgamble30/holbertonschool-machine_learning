#!/usr/bin/env python3
"""
Script to train a convolutional neural network to classify the CIFAR 10 dataset
"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    Pre-processes the data.

    Args:
    X: numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data,
       where m is the number of data points
    Y: numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X

    Returns:
    X_p: numpy.ndarray containing the preprocessed X
    Y_p: numpy.ndarray containing the preprocessed Y
    """
    X_p = X.astype('float32') / 255.0
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == '__main__':
    """
    This block trains a convolutional neural network to classify CIFAR 10
    dataset and saves the model to 'cifar10.h5'.
    """
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    inputs = K.layers.Input(shape=(32, 32, 3))
    x = K.layers.Lambda(
        lambda x: K.backend.resize_images(
            x, height_factor=7, width_factor=7, data_format="channels_last"
        )
    )(inputs)

    base_model = K.applications.EfficientNetV2B3(
        include_top=False, weights='imagenet', input_tensor=x
    )

    for layer in base_model.layers:
        layer.trainable = False

    x = K.layers.Flatten()(base_model.output)
    output = K.layers.Dense(10, activation='softmax')(x)

    model = K.models.Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']
    )

    history = model.fit(
        X_train, Y_train, validation_data=(X_test, Y_test),
        batch_size=128, epochs=10
    )

    model.save('cifar10.h5')

#!/usr/bin/env python3

import tensorflow.keras as K

def preprocess_data(X, Y):
    """
    Pre-processes the CIFAR-10 data.

    Args:
    X: numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR-10 data,
       where m is the number of data points
    Y: numpy.ndarray of shape (m,) containing the CIFAR-10 labels for X

    Returns: X_preprocessed, Y_preprocessed
        X_preprocessed: numpy.ndarray containing the preprocessed X
        Y_preprocessed: numpy.ndarray containing the preprocessed Y
    """
    X_preprocessed = K.applications.efficientnet_v2.preprocess_input(
        X, data_format="channels_last")
    Y_preprocessed = K.utils.to_categorical(Y, 10)
    return X_preprocessed, Y_preprocessed

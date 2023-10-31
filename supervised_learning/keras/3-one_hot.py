#!/usr/bin/env python3
"""converts label vector"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.

    Args:
    - labels: a label vector.
    - classes: the number of classes.

    Returns:
    - The one-hot matrix.
    """
    if classes is None:
        classes = K.backend.int_shape(labels)[-1]
    one_hot_matrix = K.utils.to_categorical(labels, num_classes=classes)
    return one_hot_matrix

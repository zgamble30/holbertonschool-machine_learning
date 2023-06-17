#!/usr/bin/env python3
"""One Hot Encode"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    - Converts a numeric label vector into a one-hot matrix.
    - Y is a numpy.ndarray with shape (m,) containing
    numeric class labels.
    - classes is the maximum number of classes found in Y.
    - Returns: a one-hot encoding of Y with shape (classes, m)
    or None on failure.
    """
    if (not isinstance(Y, np.ndarray) or len(Y) == 0 or
            not isinstance(classes, int) or classes <= np.amax(Y)):
        return None
    shap = (len(Y), classes)
    one_hot = np.zeros(shap)
    rows = np.arange(len(Y))
    one_hot[rows, Y] = 1
    # b = np.eye(classes)[Y].T
    return one_hot.T

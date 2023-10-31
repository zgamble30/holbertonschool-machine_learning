#!/usr/bin/env python3
"""
Converts a label vector into a one-hot matrix.
"""

import numpy as np


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.

    Args:
    - labels: numpy.ndarray, the label vector.
    - classes: int, the number of classes.

    Returns:
    - one-hot matrix.
    """
    if classes is None:
        classes = np.max(labels) + 1
    one_hot_matrix = np.eye(classes)[labels]
    return one_hot_matrix

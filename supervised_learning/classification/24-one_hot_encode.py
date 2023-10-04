#!/usr/bin/env python3
"""Module for converting numeric labels into a one-hot matrix."""
import numpy as np


def one_hot_encode(labels, num_classes):
    """
    Convert numeric labels into a one-hot matrix.

    Args:
        labels (numpy.ndarray): Numeric 
        label vector with shape (num_examples,).
        num_classes (int): Maximum 
        number of classes present in labels.

    Returns:
        numpy.ndarray: One-hot encoding of 
        labels with shape (num_classes, num_examples),
                      or None if the input is invalid.
    """
    # Ensure that labels is a valid numpy array with shape (num_examples,)
    if not isinstance(labels, np.ndarray) or labels.ndim != 1:
        return None

    # Ensure that num_classes is an integer
    if not isinstance(num_classes, int):
        return None

    # Get the number of examples (num_examples)
    num_examples = labels.shape[0]

    # Create an array of zeros with shape (num_classes, num_examples)
    one_hot_matrix = np.zeros((num_classes, num_examples))

    # Iterate over the label vector labels
    for example_idx, label in enumerate(labels):
        # Check if the label is within the valid range
        if label < 0 or label >= num_classes:
            return None
        # Set the corresponding element in one_hot_matrix to 1
        one_hot_matrix[label, example_idx] = 1

    # Return the resulting one-hot encoding
    return one_hot_matrix

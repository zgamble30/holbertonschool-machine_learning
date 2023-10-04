#!/usr/bin/env python3
"""Module for decoding a one-hot matrix into labels."""
import numpy as np

def one_hot_decode(one_hot):
    """
    Convert a one-hot matrix into a vector of labels.

    Args:
        one_hot (numpy.ndarray): One-hot encoded matrix with shape (num_classes, num_examples).
            num_classes (int): Maximum number of classes.
            num_examples (int): Number of examples.

    Returns:
        numpy.ndarray: Vector of labels with shape (num_examples, ), or None on failure.
    """
    # Ensure that one_hot is a valid numpy array with shape (num_classes, num_examples)
    if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None

    # Get the number of examples (num_examples)
    num_examples = one_hot.shape[1]

    # Initialize an empty label vector
    labels = np.zeros((num_examples,), dtype=int)

    # Iterate over the examples
    for i in range(num_examples):
        # Find the index of the maximum value in each example (column-wise)
        max_index = np.argmax(one_hot[:, i])

        # Set the corresponding label to the maximum index
        labels[i] = max_index

    # Return the resulting label vector
    return labels

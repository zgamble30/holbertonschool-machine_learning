#!/usr/bin/env python3
"""Module for decoding one-hot encoded matrices into labels."""
import numpy as np


def one_hot_decode(one_hot_matrix):
    """
    Convert a one-hot encoded matrix into a vector of class labels.

    Args:
        one_hot_matrix (numpy.ndarray):
            A matrix with shape (num_classes, num_examples),
            where each column represents a one-hot encoded class.

    Returns:
        numpy.ndarray: A vector with shape
        (num_examples, ) containing numeric labels,
        or None if the input is invalid.
    """
    # Check if the input is a valid numpy
    # array with shape (num_classes, num_examples)
    if not isinstance(one_hot_matrix, np.ndarray) or one_hot_matrix.ndim != 2:
        return None

    # Get the number of examples (num_examples)
    num_examples = one_hot_matrix.shape[1]

    # Initialize an empty label vector
    label_vector = np.zeros((num_examples,), dtype=int)

    # Iterate over each example
    for i in range(num_examples):
        # Find the index of the maximum value in each example (column-wise)
        max_index = np.argmax(one_hot_matrix[:, i])

        # Set the corresponding label to the maximum index
        label_vector[i] = max_index

    # Return the resulting label vector
    return label_vector


"""Example usage:
if __name__ == "__main__":
    import numpy as np

    # Sample one-hot encoded matrix (replace this with your own data)
    one_hot_data = np.array([[0, 1, 0, 0],
                             [1, 0, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])

    # Decode one-hot encoded matrix
    decoded_labels = one_hot_decode(one_hot_data)

    # Display results
    print("One-Hot Encoded Matrix:")
    print(one_hot_data)
    print("\nDecoded Labels:")
    print(decoded_labels)
"""

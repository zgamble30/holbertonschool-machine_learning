#!/usr/bin/env python3
"""Module for decoding a one-hot matrix into labels."""
import numpy as np

def one_hot_decode(one_hot):
    """
    Convert a one-hot matrix into a vector of labels.

    Args:
        one_hot (numpy.ndarray): One-hot encoded matrix with shape (num_classes, num_examples).

    Returns:
        numpy.ndarray: Vector of labels with shape (num_examples, ), or None on failure.
    """
    # Ensure that one_hot is a valid numpy array
    if not isinstance(one_hot, np.ndarray):
        return None

    # Get the number of classes and examples
    num_classes, num_examples = one_hot.shape

    # Check if the shape is valid for one-hot encoding
    if num_classes <= 0 or num_examples <= 0:
        return None

    # Find the index with the maximum value along the classes axis
    labels = np.argmax(one_hot, axis=0)

    # Return the resulting vector of labels
    return labels

"""Example usage:
if __name__ == "__main__":
    import numpy as np

    # Load data (replace this with your own data)
    label_data = np.array([5, 0, 4, 1, 9, 2, 1, 3, 1, 4])

    # Number of classes (replace this with the actual number of classes in your data)
    num_classes = 10

    # Perform one-hot encoding
    one_hot_result = oh_encode(label_data, num_classes)

    # Perform one-hot decoding
    decoded_labels = one_hot_decode(one_hot_result)

    # Print results
    print("Original Labels:")
    print(label_data)
    print("\nOne-Hot Encoding:")
    print(one_hot_result)
    print("\nDecoded Labels:")
    print(decoded_labels)
""""
#!/usr/bin/env python3
"""creates a confusion matrix"""
import numpy as np

fusion_matrix(labels, logits):
    """
    Creates a confusion matrix.
    Args:
        labels: one-hot numpy.ndarray of shape (m, classes)
            containing the correct labels for each data point.
        logits: one-hot numpy.ndarray of shape (m, classes)
            containing the predicted labels.
    Returns:
        A confusion numpy.ndarray of shape (classes, classes)
        with row indices representing the correct labels
        and column indices representing the predicted labels.
    """
    m, classes = labels.shape
    confusion_matrix = np.zeros((classes, classes))

    for i in range(m):
        true_label = np.argmax(labels[i])
        predicted_label = np.argmax(logits[i])
        confusion_matrix[true_label][predicted_label] += 1

    return confusion_matrix

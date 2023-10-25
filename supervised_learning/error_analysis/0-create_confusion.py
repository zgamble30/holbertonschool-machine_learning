#!/usr/bin/env python3
"""This script creates a confusion
matrix using two numpy arrays."""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.

    Arguments:
        labels: A numpy array of one-hot encoded shape (m, classes)
            that contains the correct labels for each data point.
        logits: A numpy array of one-hot encoded shape (m, classes)
            that contains the predicted labels.

    Returns:
        A confusion matrix in the form of a
        numpy array with shape (classes, classes).
        The row indices represent the correct
        labels, and the column indices
        represent the predicted labels.
    """

    m, classes = labels.shape
    confusion_matrix = np.zeros((classes, classes))

    for i in range(m):
        true_label = np.argmax(labels[i])
        predicted_label = np.argmax(logits[i])
        confusion_matrix[true_label][predicted_label] += 1

    return confusion_matrix

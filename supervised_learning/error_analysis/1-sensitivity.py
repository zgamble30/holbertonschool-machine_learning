#!/usr/bin/env python3
"""calculates sensitivity"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.
    Args:
        confusion: a numpy.ndarray of shape
        (classes, classes) representing the confusion matrix
        where row indices represent the correct labels
        and column indices represent the predicted labels.
    Returns:
        A numpy.ndarray of shape (classes,)
        containing the sensitivity of each class.
    """
    classes = confusion.shape[0]
    sensitivity_values = np.zeros(classes)

    for i in range(classes):
        true_positives = confusion[i, i]
        actual_positives = np.sum(confusion[i, :])
        sensitivity_values[i] = true_positives / actual_positives

    return sensitivity_values

#!/usr/bin/env python3
"""calculates precision"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.
    Args:
        confusion: a numpy.ndarray of shape (classes,
        classes) representing the confusion matrix
        where row indices represent the correct labels
        and column indices represent the predicted labels.
    Returns:
        A numpy.ndarray of shape (classes,)
        containing the precision of each class.
    """
    classes = confusion.shape[0]
    precision_values = np.zeros(classes)

    for i in range(classes):
        true_positives = confusion[i, i]
        predicted_positives = np.sum(confusion[:, i])
        if predicted_positives == 0:
            precision_values[i] = 0.0
        else:
            precision_values[i] = true_positives / predicted_positives

    return precision_values

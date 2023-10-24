#!/usr/bin/env python3
"""calculates sensitivity"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.
    Args:
        confusion: numpy.ndarray of shape (classes, classes)
            where row indices represent the correct
            labels and column indices represent the predicted labels.
        classes: the number of classes.
    Returns:
        A numpy.ndarray of shape (classes,)
        containing the sensitivity of each class.
    """
    true_positives = np.diag(confusion)
    actual_positives = np.sum(confusion, axis=1)
    
    sensitivity_values = true_positives / actual_positives

    return sensitivity_values

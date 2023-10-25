#!/usr/bin/env python3
"""calculates specificity"""
import numpy as np

def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.

    Args:
        confusion: a numpy.ndarray of shape (classes, classes) where row indices
                   represent the correct labels and column indices represent the
                   predicted labels.

    Returns:
        A numpy.ndarray of shape (classes,) containing the specificity of each class.
    """
    classes = confusion.shape[0]
    spec = np.zeros(classes)

    for i in range(classes):
        true_negative = np.sum(np.delete(np.delete(confusion, i, axis=0), i, axis=1))
        actual_negative = np.sum(np.delete(confusion[i, :], i))
        spec[i] = true_negative / (true_negative + actual_negative)

    return spec

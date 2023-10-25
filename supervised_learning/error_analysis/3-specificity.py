#!/usr/bin/env python3
"""calculates specificity"""
import numpy as np


def specificity(confusion):
    """
    Calculate the specificity for each class in a confusion matrix.

    Args:
        confusion: a numpy.ndarray of shape
        (classes, classes) where row indices
                   represent the correct labels
                   and column indices represent the
                   predicted labels.

    Returns:
        A numpy.ndarray of shape (classes,)
        containing the specificity of each class.
    """
    classes = confusion.shape[0]
    spec = np.zeros(classes)

    for i in range(classes):
        true_negative = (
            np.sum(confusion) -
            np.sum(confusion[i, :]) -
            np.sum(confusion[:, i]) +
            confusion[i, i]
            )

        actual_negative = np.sum(confusion[:, i]) - confusion[i, i]
        spec[i] = true_negative / (true_negative + actual_negative)

    return spec

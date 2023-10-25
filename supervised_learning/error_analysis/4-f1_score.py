#!/usr/bin/env python3
"""F1 Score Calculation"""

import numpy as np


# Import the previously implemented sensitivity and precision functions
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision

def f1_score(confusion):
    """
    Calculate the F1 score for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): Confusion matrix of shape (classes, classes).

    Returns:
        numpy.ndarray: F1 scores for each class.
    """
    classes = confusion.shape[0]
    f1_scores = np.zeros(classes)

    for i in range(classes):
        sens = sensitivity(confusion)
        prec = precision(confusion, i)

        f1_scores[i] = 2 * (sens[i] * prec) / (sens[i] + prec)

    return f1_scores

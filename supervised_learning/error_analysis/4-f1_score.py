#!/usr/bin/env python3
"""F1 Score Calculation"""

import numpy as np

# Import the sensitivity and precision functions using __import__
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculate the F1 score for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray):
        Confusion matrix of shape (classes, classes).

    Returns:
        numpy.ndarray: F1 scores for each class.
    """
    classes = confusion.shape[0]
    f1_scores = np.zeros(classes)

    for i in range(classes):
        sens = sensitivity(confusion)
        prec = precision(confusion)

        f1_scores[i] = 2 * (sens[i] * prec[i]) / (sens[i] + prec[i])

    return f1_scores

#!/usr/bin/env python3
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Create a confusion matrix.

    Args:
        labels: A one-hot numpy.ndarray of
        shape (m, classes) containing correct labels.
        logits: A one-hot numpy.ndarray of shape (m, classes)
        containing predicted labels.

    Returns:
        A confusion numpy.ndarray of shape (classes, classes).
    """
    m, classes = labels.shape
    confusion = np.zeros((classes, classes), dtype=int)

    for i in range(m):
        true_label = np.argmax(labels[i])
        predicted_label = np.argmax(logits[i])
        confusion[true_label][predicted_label] += 1

    return confusion

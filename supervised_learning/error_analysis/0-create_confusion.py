#!/usr/bin/env python3
"""create confusion"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Create a confusion matrix from
    one-hot encoded labels and predicted logits.

    Args:
        labels (numpy.ndarray): One-hot
        encoded labels (shape: [m, classes]).
        logits (numpy.ndarray):
        Predicted logits (shape: [m, classes]).

    Returns:
        numpy.ndarray: Confusion
        matrix (shape: [classes, classes]).
    """
    m, classes = labels.shape
    confusion = np.zeros((classes, classes), dtype=int)
    # Initialize as integer values

    for i in range(m):
        true_label = np.argmax(labels[i])
        predicted_label = np.argmax(logits[i])
        confusion[true_label][predicted_label] += 1

    return confusion

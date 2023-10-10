#!/usr/bin/env python3
"""
Module containing a function to calculate the accuracy of a prediction.
"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction.

    Args:
        y (tf.Tensor): Placeholder for the labels of the input data.
        y_pred (tf.Tensor): Tensor containing the network’s predictions.

    Returns:
        tf.Tensor: A tensor containing the decimal accuracy of the prediction.
    """
    # Compare predicted labels with true labels
    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='Mean')

    return accuracy

#!/usr/bin/env python3
"""
Module containing a function to calculate the softmax cross-entropy loss.
"""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss of a prediction.

    Args:
        y (tf.Tensor): Placeholder for the labels of the input data.
        y_pred (tf.Tensor): Tensor containing the network’s predictions.

    Returns:
        tf.Tensor: A tensor containing the loss of the prediction.
    """
    # Compute softmax cross-entropy loss
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred, name='softmax_cross_entropy_loss')

    return loss

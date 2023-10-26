#!/usr/bin/env python3
"""Calculates the cost of a neural
network with L2 regularization using TensorFlow"""

import tensorflow as tf


def l2_reg_cost(cost):
    """
    Calculate the cost of a neural
    network with L2 regularization.

    Args:
        cost: A tensor containing the cost
        of the network without L2 regularization.

    Returns:
        A tensor containing the cost of the
        network accounting for L2 regularization.
    """
    l2_loss = tf.losses.get_regularization_loss(scope=None)
    cost_with_l2 = cost + l2_loss
    return cost_with_l2

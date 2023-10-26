#!/usr/bin/env python3
"""
This function calculates the cost of a neural network with L2 regularization.
"""

import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """
    Calculate the cost of a neural network with L2 regularization.

    Args:
        cost: A tensor containing the cost of the network without L2 regularization.

    Returns:
        A tensor containing the cost of the network accounting for L2 regularization.
    """
    # Get the L2 regularization losses from TensorFlow.
    l2_loss = tf.losses.get_regularization_losses()

    # Combine the original cost and L2 regularization losses to get the final cost.
    cost_with_l2 = cost + l2_loss

    return cost_with_l2

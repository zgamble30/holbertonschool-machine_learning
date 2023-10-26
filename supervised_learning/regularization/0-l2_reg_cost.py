#!/usr/bin/env python3
"""calculates the cost of a neural network with L2 regularization"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculate the cost of a neural network with L2 regularization.

    Args:
        cost: the cost of the network without L2 regularization.
        lambtha: the regularization parameter.
        weights: a dictionary of the weights
        and biases (numpy.ndarrays) of the neural network.
        L: the number of layers in the neural network.
        m: the number of data points used.

    Returns:
        The cost of the network accounting for L2 regularization.
    """
    l2_reg = 0  # Initialize L2 regularization term.

    for i in range(1, L + 1):
        W = weights['W' + str(i)]  # Get the weight matrix for layer i.
        l2_reg += np.linalg.norm(W)

    l2_reg = (lambtha / (2 * m)) * l2_reg
    cost += l2_reg  # Add the regularization term to the cost.

    return cost

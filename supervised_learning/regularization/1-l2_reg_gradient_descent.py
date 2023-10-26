#!/usr/bin/env python3
"""
Calculates the gradient of a neural network with L2 regularization
using gradient descent
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient descent
    with L2 regularization.

    Args:
        Y: One-hot numpy.ndarray of shape (classes, m)
        weights: Dictionary of the weights and biases of the neural network
        cache: Dictionary of the outputs of each layer of the neural network
        alpha: Learning rate
        lambtha: L2 regularization parameter
        L: Number of layers of the network

    Returns:
        None (weights and biases are updated in place)
    """
    m = Y.shape[1]
    for i in range(L, 0, -1):
        A = cache["A" + str(i)]
        if i == L:
            dZ = A - Y
        else:
            dZ = np.matmul(weights["W" + str(i + 1)].T, dZ) * (1 - A**2)
        dW = (1 / m) * np.matmul(dZ, cache["A" + str(i - 1)].T) + (lambtha / m) * weights["W" + str(i)]
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        weights["W" + str(i)] = weights["W" + str(i)] - alpha * dW
        weights["b" + str(i)] = weights["b" + str(i)] - alpha * db

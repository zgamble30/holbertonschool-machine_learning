#!/usr/bin/env python3
"""Gradient Descent with L2 Regularization"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Update the weights and biases of a neural network using gradient descent
    with L2 regularization.

    Args:
        Y: One-hot numpy.ndarray of shape (classes, m)
        that contains the correct labels for the data.
        weights: Dictionary of the weights and biases of the neural network.
        cache: Dictionary of the outputs of each layer of the neural network.
        alpha: Learning rate.
        lambtha: L2 regularization parameter.
        L: Number of layers of the network.

    Notes:
        The neural network uses tanh activations on each layer
        except the last, which uses a softmax activation.

    Returns:
        None (weights and biases are updated in place).
    """
    m = Y.shape[1]
    dAPrevLayer = None
    # Initialize the derivative with respect to the previous layer's activation
    
    for layer in range(L, 0, -1):
        A_cur = cache["A{}".format(layer)]  # Current layer's activation
        A_prev = cache["A{}".format(layer - 1)]  # Previous layer's activation
        if layer == L:
            dz = A_cur - Y  # Compute the initial delta for the output layer
        else:
            dz = dAPrevLayer * (1 - np.square(A_cur))
            # Compute delta for hidden layers
        W = weights["W{}".format(layer)]  # Weights for the current layer
        l2 = (lambtha / m) * W  # L2 regularization term
        dW = np.matmul(dz, A_prev.T) / m + l2
        # Compute the gradient of the weights with L2 regularization
        db = np.sum(dz, axis=1, keepdims=True) / m
        # Compute the gradient of the biases
        dAPrevLayer = np.matmul(W.T, dz)
        # Compute the derivative with respect to the previous layer's activation
        # Update the weights and biases using the gradient descent rule
        weights["W{}".format(layer)] -= alpha * dW
        weights["b{}".format(layer)] -= alpha * db

#!/usr/bin/env python3
"""Gradient Descent with Dropout"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Update the weights of a neural network with
    Dropout regularization using gradient descent.

    Args:
        Y (numpy.ndarray): One-hot matrix of
        shape (classes, m) containing the correct labels for the data.
        weights (dict): Dictionary of the
        weights and biases of the neural network.
        cache (dict): Dictionary of the
        outputs and dropout masks of each layer of the neural network.
        alpha (float): The learning rate for gradient descent.
        keep_prob (float): The probability that a node will be kept.
        L (int): The number of layers in the network.

    Returns:
        None (weights are updated in place).
    """

    m = Y.shape[1]
    dZ_output = (cache["A{}".format(L)] - Y)

    for layer in range(L, 0, -1):
        A_prev = cache["A{}".format(layer - 1)]
        dW = np.dot(dZ_output, A_prev.T) / m
        db = np.sum(dZ_output, axis=1, keepdims=True) / m

        if layer > 1:
            dZ_output = np.dot(weights["W{}".format(layer)].T, dZ_output)
            dZ_output = dZ_output * (1 - np.power(A_prev, 2))
            dZ_output *= cache["D{}".format(layer - 1)]
            dZ_output /= keep_prob

        # Update weights and biases
        weights["W{}".format(layer)] -= alpha * dW
        weights["b{}".format(layer)] -= alpha * db

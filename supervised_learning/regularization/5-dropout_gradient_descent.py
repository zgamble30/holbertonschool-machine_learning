#!/usr/bin/env python3
"""Gradient Descent with Dropout"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a
    neural network with Dropout regularization
    using gradient descent.

    Args:
        Y: The ground truth labels as a
        numpy.ndarray of shape (output_size, num_samples).
        weights: Dictionary containing weights and biases for each layer.
        cache: Dictionary containing
        the outputs of each layer and dropout masks.
        alpha: The learning rate for gradient descent.
        keep_prob: The probability that a node will be kept.
        L: The number of layers in the neural network.

    Returns:
        Updated weights and biases.
    """
    m = Y.shape[1]
    back = {}

    for index in range(L, 0, -1):
        A_current = cache[f"A{index}"]
    
        if index == L:
            back[f"dz{index}"] = (A_current - Y)
            dz = back[f"dz{index}"]
        else:
            dz_next = back[f"dz{index + 1}"]
            W_next = weights[f"W{index + 1}"]

            # Transpose one of the matrices to ensure compatibility
            dz = np.dot(W_next.T, dz_next) * (A_current * (1 - A_current))
            dz *= cache[f"D{index}"]
            dz /= keep_prob

        dW = (1 / m) * np.dot(dz, cache[f"A{index - 1}"].T)
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

        weights[f"W{index}"] -= alpha * dW
        weights[f"b{index}"] -= alpha * db

    return weights

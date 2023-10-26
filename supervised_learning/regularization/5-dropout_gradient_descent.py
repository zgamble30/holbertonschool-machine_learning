#!/usr/bin/env python3
"""Gradient Descent with Dropout"""

import numpy as np


def update_weights(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
    using gradient descent.

    Args:
        Y: The ground truth labels as a numpy.ndarray of shape (output_size, num_samples).
        weights: Dictionary containing weights and biases for each layer.
        cache: Dictionary containing the outputs of each layer and dropout masks.
        alpha: The learning rate for gradient descent.
        keep_prob: The probability that a node will be kept.
        L: The number of layers in the neural network.

    Returns:
        Updated weights and biases.
    """
    m = Y.shape[1]
    backward_cache = {}

    for index in range(L, 0, -1):
        A_current = cache[f"A{index - 1}"]
        A_next = cache[f"A{index}"]
        W_current = weights[f"W{index}"]
        dz = None

        if index == L:
            # Output layer
            dz = A_next - Y
        else:
            dz_next = backward_cache[f"dz{index + 1}"]
            W_next = weights[f"W{index + 1}"]

            dz = np.dot(W_next.T, dz_next) * (A_current * (1 - A_current))
            dz *= cache[f"D{index}"]
            dz /= keep_prob

        dW = np.dot(dz, A_current.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m

        weights[f"W{index}"] -= alpha * dW
        weights[f"b{index}"] -= alpha * db

        backward_cache[f"dz{index}"] = dz

    return weights

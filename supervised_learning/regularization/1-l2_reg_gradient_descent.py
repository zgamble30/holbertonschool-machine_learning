#!/usr/bin/env python3
"""
Updates the weights and biases of a neural
network using gradient descent with L2 regularization.
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Update the weights and biases of a neural network
    using gradient descent with L2 regularization.

    Args:
        Y: a one-hot numpy.ndarray of shape (classes, m)
        that contains the correct labels for the data.
        weights: a dictionary of the weights and biases of the neural network.
        cache: a dictionary of the outputs of each layer of the neural network.
        alpha: the learning rate.
        lambtha: the L2 regularization parameter.
        L: the number of layers of the network.

    The neural network uses tanh activations on each
    layer except the last, which uses a softmax activation.
    The weights and biases of the network should be updated in place.
    """
    m = Y.shape[1]

    for i in range(L, 0, -1):
        A = cache['A' + str(i)]
        A_prev = cache['A' + str(i - 1)]

        if i == L:
            dZ = A - Y
        else:
            dZ = np.dot(weights['W' + str(i + 1)].T, dZ) * (1 - A ** 2)

        dW = np.dot(dZ, A_prev.T) / m + (lambtha / m) * weights['W' + str(i)]
        db = np.sum(dZ, axis=1, keepdims=True) / m
        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db

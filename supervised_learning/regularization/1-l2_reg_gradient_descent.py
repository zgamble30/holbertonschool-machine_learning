#!/usr/bin/env python3
"""Gradient Descent with L2 Regularization"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache,
                            learning_rate, lambda_param, num_layers):

    """
    Update the weights and biases
    of a neural network using gradient descent
    with L2 regularization.

    Args:
        Y: One-hot numpy.ndarray of shape
        (classes, m) containing the correct labels for the data.
        weights: Dictionary of the weights and biases of the neural network.
        cache: Dictionary of the layer
        activations for each layer of the neural network.
        learning_rate: Learning rate.
        lambda_param: L2 regularization parameter.
        num_layers: Number of layers in the network.

    Notes:
        The neural network uses tanh activations
        on each layer except the last, which uses a softmax activation.

    Returns:
        None (weights and biases are updated in place).
    """
    num_samples = Y.shape[1]
    dA_previous_layer = None
    """initialize the derivative with
    respect to the previous layer's activation"""

    for layer in range(num_layers, 0, -1):
        A_current_layer = cache["A{}".format(layer)]
        # Current layer's activation
        A_previous_layer = cache["A{}".format(layer - 1)]
        # Previous layer's activation

        if layer == num_layers:
            dZ = A_current_layer - Y
            # Compute the initial delta for the output layer
        else:
            dZ = dA_previous_layer * (1 - np.square(A_current_layer))
            # Compute delta for hidden layers

        W_current_layer = weights["W{}".format(layer)]
        # Weights for the current layer
        l2_regularization_term = (lambda_param / num_samples) * W_current_layer
        # L2 regularization term
        dW = np.matmul(dZ, A_previous_layer.T) / num_samples
        dW += l2_regularization_term

        # Gradient of the weights with L2 regularization
        db = np.sum(dZ, axis=1, keepdims=True) / num_samples
        # Gradient of the biases
        dA_previous_layer = np.matmul(W_current_layer.T, dZ)
        # Derivative with respect to the previous layer's activation

        # Update the weights and biases using the gradient descent rule
        weights["W{}".format(layer)] -= learning_rate * dW
        weights["b{}".format(layer)] -= learning_rate * db

#!/usr/bin/env python3
"""conducts forward propagation using Dropout"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conduct forward propagation using Dropout.
    
    Args:
        X: numpy.ndarray of shape (nx, m)
        containing the input data for the network.
        weights: a dictionary of the weights and biases of the neural network.
        L: the number of layers in the network.
        keep_prob: the probability that a node will be kept.

    Returns:
        A dictionary containing the outputs of each
        layer and the dropout mask used on each layer.
    """
    m = X.shape[1]
    cache = {'A0': X}
    dropout_masks = {}

    for layer in range(1, L + 1):
        Z = np.dot(weights[f'W{layer}'], cache[f'A{layer - 1}']) + weights[f'b{layer}']
        if layer != L:
            A = np.tanh(Z)
            # Apply dropout
            dropout_mask = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            A *= dropout_mask
            A /= keep_prob
            dropout_masks[f'D{layer}'] = dropout_mask
        else:
            exp_Z = np.exp(Z)
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        cache[f'Z{layer}'] = Z
        cache[f'A{layer}'] = A

    return cache, dropout_masks

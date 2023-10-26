#!/usr/bin/env python3
"""Forward Propagation with Dropout"""

import numpy as np


def dropout_forward_prop(X, weights, num_layers, keep_prob):
    """
    Conducts forward propagation using Dropout.

    Args:
        X: Input data as a numpy.ndarray of shape (input_size, num_samples).
        weights: Dictionary containing weights and biases for each layer.
        num_layers: The number of layers in the neural network.
        keep_prob: The probability that a node will be kept.

    Returns:
        A dictionary containing the outputs of
        each layer and the dropout masks used for each layer.
    """
    layer_cache = {}
    layer_cache["A0"] = X

    for layer in range(1, num_layers + 1):
        weight_matrix = weights[f"W{layer}"]
        previous_layer_activation = layer_cache[f"A{layer - 1}"]
        bias = weights[f"b{layer}"]
        weighted_sum = np.matmul(weight_matrix, previous_layer_activation)
        weighted_sum = weighted_sum + bias


        if layer == num_layers:
            softmax_numerator = np.exp(weighted_sum)
            softmax_denominator = np.sum(softmax_numerator, axis=0)
            softmax_activation = softmax_numerator / softmax_denominator

            layer_cache[f"A{layer}"] = softmax_activation
        else:
            numerator = np.exp(weighted_sum) - np.exp(-weighted_sum)
            denominator = np.exp(weighted_sum) + np.exp(-weighted_sum)
            tanh_activation = numerator / denominator
            num_rows, num_cols = tanh_activation.shape
            dropout_mask = np.random.rand(num_rows, num_cols) < keep_prob
            layer_cache[f"D{layer}"] = dropout_mask.astype(int)
            tanh_activation *= dropout_mask
            tanh_activation /= keep_prob
            layer_cache[f"A{layer}"] = tanh_activation

    return layer_cache

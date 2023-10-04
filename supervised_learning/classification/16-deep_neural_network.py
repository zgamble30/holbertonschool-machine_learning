#!/usr/bin/env python3
"""Deep neural network module"""

import numpy as np

class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification."""

    def __init__(self, nx, layers):
        """Class constructor"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) is not list or not layers or not all(isinstance(i, int) and i > 0 for i in layers):
            raise TypeError('layers must be a list of positive integers')

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for l in range(1, self.L + 1):
            key_W = 'W' + str(l)
            key_b = 'b' + str(l)
            if l == 1:
                self.weights[key_W] = np.random.randn(layers[0], nx) * np.sqrt(2 / nx)
            else:
                self.weights[key_W] = np.random.randn(layers[l - 1], layers[l]) * np.sqrt(2 / layers[l - 1])
            self.weights[key_b] = np.zeros((layers[l], 1))

    @property
    def L(self):
        """Getter method for L (number of layers)."""
        return self.__L

    @property
    def cache(self):
        """Getter method for cache (intermediary values)."""
        return self.__cache

    @property
    def weights(self):
        """Getter method for weights (weights and biases)."""
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network."""
        self.__cache['A0'] = X
        for l in range(1, self.L + 1):
            key_W = 'W' + str(l)
            key_b = 'b' + str(l)
            key_A = 'A' + str(l - 1)
            key_new_A = 'A' + str(l)
            Z = np.dot(self.weights[key_W], self.cache[key_A]) + self.weights[key_b]
            self.cache[key_new_A] = 1 / (1 + np.exp(-Z))
        return self.cache[key_new_A], self.cache

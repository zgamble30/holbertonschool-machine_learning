#!/usr/bin/env python3
"""Deep Neural Network class"""

import numpy as np


class DeepNeuralNetwork:
    """Class defining a deep neural network for binary classification."""

    @staticmethod
    def initialize_weights(nx, layers):
        """
        Initializes weights using the He et al. method.

        Args:
            nx (int): Number of input features.
            layers (list): List representing the number of nodes in each layer.

        Returns:
            dict: Dictionary containing initialized weights and biases.
        """
        weights = dict()
        for i in range(1, len(layers) + 1):
            # Check if nodes is an integer and positive
            if not isinstance(layers[i - 1], int) or layers[i - 1] < 1:
                raise TypeError('layers must be a list of positive integers')
            
            prev_nodes = nx if i == 1 else layers[i - 2]
            weight_matrix = np.random.randn(layers[i - 1], prev_nodes)
            weight_scaling = np.sqrt(2 / prev_nodes)
            weights.update({
                'b' + str(i): np.zeros((layers[i - 1], 1)),
                'W' + str(i): weight_matrix * weight_scaling
            })
        return weights

    def __init__(self, nx, layers):
        """
        Class constructor.

        Args:
            nx (int): Number of input features.
            layers (list): List representing the number of nodes in each layer.
        """
        # Check if nx is an integer
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')

        # Check if nx is a positive integer
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        # Check if layers is a list
        if not isinstance(layers, list):
            raise TypeError('layers must be a list of positive integers')

        # Check if elements in layers are positive integers
        if not all(isinstance(nodes, int) and nodes > 0 for nodes in layers):
            raise TypeError('layers must be a list of positive integers')

        # Set private instance attributes
        self.__L = len(layers)
        self.__cache = dict()
        self.__weights = self.initialize_weights(nx, layers)

    @property
    def L(self):
        """Getter for the number of layers"""
        return self.__L

    @property
    def cache(self):
        """Getter for the cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for the weights"""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).

        Returns:
            Tuple: Output of the neural network and the cache.
        """
        self.__cache['A0'] = X

        for i in range(1, self.L + 1):
            W_key = 'W' + str(i)
            b_key = 'b' + str(i)
            A_key = 'A' + str(i)

            Z = np.dot(self.weights[W_key], self.cache['A' + str(i - 1)]) + self.weights[b_key]
            A = 1 / (1 + np.exp(-Z))

            self.__cache[A_key] = A

        return self.cache['A' + str(self.L)], self.cache

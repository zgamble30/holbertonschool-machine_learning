#!/usr/bin/env python3
"""Deep Neural Network class"""

import numpy as np


class DeepNeuralNetwork:
    """Neural Network for Binary Classification"""

    @staticmethod
    def initialize_weights(nx, layers):
        """
        Initialize weights.

        Args:
            nx (int): Number of input features.
            layers (list): List representing the number of nodes in each layer.

        Returns:
            dict: Dictionary containing initialized weights and biases.
        """
        weights = dict()
        for i, nodes in enumerate(layers, start=1):
            if not isinstance(nodes, int) or nodes < 1:
                raise TypeError('layers must be a list of positive integers')
            prev_nodes = layers[i - 2] if i > 1 else nx
            weight_matrix = np.random.randn(nodes, prev_nodes)
            weight_scaling = np.sqrt(2 / prev_nodes)
            weights.update({
                'b' + str(i): np.zeros((nodes, 1)),
                'W' + str(i): weight_matrix * weight_scaling
            })
        return weights

    def __init__(self, nx, layers):
        """
        Neural Network constructor.

        Args:
            nx (int): Number of input features.
            layers (list): List representing the number of nodes in each layer.
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(layers, list) or not all(isinstance(i, int) for i in layers) or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')

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

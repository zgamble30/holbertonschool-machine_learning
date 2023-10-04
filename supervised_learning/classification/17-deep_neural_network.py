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

    @L.setter
    def L(self, value):
        """Setter for the number of layers"""
        print("Cannot set L directly. Use the constructor with the desired number of layers.")


""" Testing the class
if __name__ == "__main__":
    lib_train = np.load('../data/Binary_Train.npz')
    X_3D, Y = lib_train['X'], lib_train['Y']
    X = X_3D.reshape((X_3D.shape[0], -1)).T

    np.random.seed(0)
    deep = DeepNeuralNetwork(X.shape[0], [5, 3, 1])
    print(deep.cache)
    print(deep.weights)
    print(deep.L)
    deep.L = 10  # This line should print the error message
"""
#!/usr/bin/env python3
"""This is a deep neural network class"""

import numpy as np


class DeepNeuralNetwork:
    """Class defining a deep neural network for binary classification."""

    @staticmethod
    def initialize_weights(nx, layers):
        """
        Initializes weights 

        Args:
            nx (int): Number of input features.
            layers (list): List representing the number of nodes in each layer.

        Returns:
            dict: Dictionary containing initialized weights and biases.
        """
        weights = dict()
        for i in range(len(layers)):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError('layers must be a list of positive integers')
            prev_layer = layers[i - 1] if i > 0 else nx
            w_part1 = np.random.randn(layers[i], prev_layer)
            w_part2 = np.sqrt(2 / prev_layer)
            weights.update({
                'b' + str(i + 1): np.zeros((layers[i], 1)),
                'W' + str(i + 1): w_part1 * w_part2
            })
        return weights

    def __init__(self, nx, layers):
        """
        Class constructor.

        Args:
            nx (int): Number of input features.
            layers (list): List representing the number of nodes in each layer.
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')

        self.__L = len(layers)
        self.__cache = dict()
        self.__weights = self.initialize_weights(nx, layers)

    @property
    def L(self):
        """Getter for the number of layers."""
        return self.__L

    @property
    def cache(self):
        """Getter for the cache."""
        return self.__cache

    @property
    def weights(self):
        """Getter for the weights."""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).

        Returns:
            tuple: Output of the neural network (A) and the cache dictionary.
        """
        self.__cache['A0'] = X
        for i in range(1, self.__L + 1):
            W_key, b_key = 'W' + str(i), 'b' + str(i)
            A_key = 'A' + str(i)
            Z = np.dot(self.__weights[W_key], self.__cache['A' + str(i - 1)]) + self.__weights[b_key]
            activation = 1 / (1 + np.exp(-Z))
            self.__cache[A_key] = activation

        return self.__cache['A' + str(self.__L)], self.__cache

"""Testing the class
if __name__ == "__main__":
    lib_train = np.load('../data/Binary_Train.npz')
    X_3D, Y = lib_train['X'], lib_train['Y']
    X = X_3D.reshape((X_3D.shape[0], -1)).T

    np.random.seed(0)
    deep = DeepNeuralNetwork(X.shape[0], [5, 3, 1])
    deep._DeepNeuralNetwork__weights['b1'] = np.ones((5, 1))
    deep._DeepNeuralNetwork__weights['b2'] = np.ones((3, 1))
    deep._DeepNeuralNetwork__weights['b3'] = np.ones((1, 1))
    A, cache = deep.forward_prop(X)
    print(A)
    print(cache)
    print(cache is deep.cache)
    print(A is cache['A3'])
"""

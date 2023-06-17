#!/usr/bin/env python3
"""Deep Neural Network"""
import numpy as np


class DeepNeuralNetwork():
    """Deep Neural Network"""

    def __init__(self, nx, layers):
        """
        - Defines a deep neural network performing binary classification
        - nx is the number of input features.
        - layers is a list representing the number of nodes in each
        layer of the network
        """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(layers) != list:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(0, self.L):
            if layers[i] < 0:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                self.__weights["W" + str(i + 1)] = np.random.randn(
                            layers[i], nx)*np.sqrt(2/(nx))
                self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))
            else:
                self.__weights["W" + str(i + 1)] = np.random.randn(
                            layers[i], layers[i-1]) * np.sqrt(2/(layers[i-1]))
                self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """ Get the L (amount of layers)
        """
        return self.__L

    @property
    def cache(self):
        """ Get the cache
        """
        return self.__cache

    @property
    def weights(self):
        """ Get the weights
        """
        return self.__weights

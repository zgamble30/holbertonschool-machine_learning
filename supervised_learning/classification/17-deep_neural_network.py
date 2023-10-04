#!/usr/bin/env python3
import numpy as np


class NeuralNetwork:
    """
    Defines a neural network for binary classification.
    """
    def __init__(self, input_size, layer_sizes):
        """
        Initializes the neural network.

        Parameters:
        - input_size (int): Number of input features.
        - layer_sizes (list): List of positive integers representing the
          number of nodes in each hidden layer.

        Raises:
        - TypeError: If input_size is not an integer, or if layer_sizes is not
          a list of positive integers.
        - ValueError: If input_size is not a positive integer.
        """
        if not isinstance(input_size, int):
            raise TypeError("input_size must be an integer")
        if input_size < 1:
            raise ValueError("input_size must be a positive integer")
        if not isinstance(layer_sizes, list) or len(layer_sizes) == 0:
            raise TypeError("layer_sizes must be a list of positive integers")
        if not all(map(lambda x: x > 0 and isinstance(x, int), layer_sizes)):
            raise TypeError("layer_sizes must be a list of positive integers")

        self.__input_size = input_size
        self.__layer_sizes = layer_sizes
        self.__num_layers = len(layer_sizes)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__num_layers):
            if i == 0:
                self.__weights['W' + str(i + 1)] = np.random.randn(layer_sizes[i], input_size) * np.sqrt(2 / input_size)
            else:
                self.__weights['W' + str(i + 1)] = np.random.randn(layer_sizes[i], layer_sizes[i - 1]) * np.sqrt(2 / layer_sizes[i - 1])
            self.__weights['b' + str(i + 1)] = np.zeros((layer_sizes[i], 1))

    @property
    def num_layers(self):
        """
        Returns the number of layers in the neural network.
        """
        return self.__num_layers

    @property
    def cache(self):
        """
        Returns the cache dictionary containing intermediary values.
        """
        return self.__cache

    @property
    def weights(self):
        """
        Returns the weights dictionary containing weights and biases.
        """
        return self.__weights

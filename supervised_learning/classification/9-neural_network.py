#!/usr/bin/env python3
"""Neural Network"""

import numpy as np


class NeuralNetwork:
    """Defines a neural network with one hidden layer
    performing binary classification"""

    def __init__(self, nx, nodes):
        """
        Class constructor.

        Args:
            - nx (int): Number of input features.
            - nodes (int): Number of nodes in the hidden layer.

        Raises:
            - TypeError: If nx or nodes is not an integer.
            - ValueError: If nx or nodes is less than 1.
        """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        # Initialize weights and biases for the hidden layer
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        # Initialize weights and biases for the output neuron
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter method for W1 (weights of hidden layer)."""
        return self.__W1

    @property
    def b1(self):
        """Getter method for b1 (biases of hidden layer)."""
        return self.__b1

    @property
    def A1(self):
        """Getter method for A1 (activated output of hidden layer)."""
        return self.__A1

    @property
    def W2(self):
        """Getter method for W2 (weights of output neuron)."""
        return self.__W2

    @property
    def b2(self):
        """Getter method for b2 (bias of output neuron)."""
        return self.__b2

    @property
    def A2(self):
        """Getter method for A2 (activated output of output neuron)."""
        return self.__A2

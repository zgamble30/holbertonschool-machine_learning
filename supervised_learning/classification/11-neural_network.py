#!/usr/bin/env python3
"""This is a neural network"""

import numpy as np


class NeuralNetwork:
    """Defines a binary classification neural network with a single hidden layer."""

    def __init__(self, nx, nodes):
        """Class constructor"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.

        Args:
            - X (numpy.ndarray): Input data with shape (nx, m).

        Returns:
            - Tuple of numpy.ndarray: Activations of hidden and output layers.
        """
        # Hidden layer calculation
        Z1 = np.dot(self.W1, X) + self.b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        # Output layer calculation
        Z2 = np.dot(self.W2, self.A1) + self.b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.A1, self.A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Args:
            - Y (numpy.ndarray): Correct labels with shape (1, m).
            - A (numpy.ndarray): Activated output with shape (1, m).

        Returns:
            - float: Cost of the model.
        """
        m = Y.shape[1]
        epsilon = 1.0000001
        cost = -(1 / m) * np.sum(Y * np.log(A + 1e-8) +
        (1 - Y) * np.log(1 - A + 1e-8))
        return cost

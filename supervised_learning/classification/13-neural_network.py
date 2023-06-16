#!/usr/bin/env python3
"""
Module that defines a neural network with one hidden layer
performing binary classification
"""

import numpy as np


class NeuralNetwork:
    """
    NeuralNetwork class
    """

    def __init__(self, nx, nodes):
        """
        Constructor method
        """

        self.nx = nx
        self.nodes = nodes

        self.__W1 = np.random.randn(self.nodes, self.nx)
        self.__b1 = np.zeros((self.nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.randn(1, self.nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        Getter method for W1
        """
        return self.__W1

    @property
    def b1(self):
        """
        Getter method for b1
        """
        return self.__b1

    @property
    def A1(self):
        """
        Getter method for A1
        """
        return self.__A1

    @property
    def W2(self):
        """
        Getter method for W2
        """
        return self.__W2

    @property
    def b2(self):
        """
        Getter method for b2
        """
        return self.__b2

    @property
    def A2(self):
        """
        Getter method for A2
        """
        return self.__A2

    def forward_prop(self, X):
        """
        Method that calculates the forward propagation
        """

        self.__A1 = 1 / (1 + np.exp(-(
            np.matmul(self.__W1, X) + self.__b1)))
        self.__A2 = 1 / (1 + np.exp(-(
            np.matmul(self.__W2, self.__A1) + self.__b2)))

        return self.__A1, self.__A2

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        """

        m = X.shape[1]

        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dZ1 = np.dot(self.__W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1

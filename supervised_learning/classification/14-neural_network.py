#!/usr/bin/env python3

"""
Module that defines a neural network with one hidden layer performing binary classification.
"""

import numpy as np


class NeuralNetwork:
    """
    NeuralNetwork class.
    """

    def __init__(self, nx, nodes):
        """
        Constructor method.
        """
        self.nx = nx
        self.nodes = nodes

        # Initialize weights and biases
        self.__W1 = np.random.randn(self.nodes, self.nx)
        self.__b1 = np.zeros((self.nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.randn(1, self.nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        Getter method for W1.
        """
        return self.__W1

    @property
    def b1(self):
        """
        Getter method for b1.
        """
        return self.__b1

    @property
    def A1(self):
        """
        Getter method for A1.
        """
        return self.__A1

    @property
    def W2(self):
        """
        Getter method for W2.
        """
        return self.__W2

    @property
    def b2(self):
        """
        Getter method for b2.
        """
        return self.__b2

    @property
    def A2(self):
        """
        Getter method for A2.
        """
        return self.__A2

    def forward_prop(self, X):
        """
        Method that calculates the forward propagation.
        """
        self.__A1 = 1 / (1 + np.exp(-(np.matmul(self.__W1, X) + self.__b1)))
        self.__A2 = 1 / (1 + np.exp(-(np.matmul(self.__W2, self.__A1) + self.__b2)))
        return self.__A1, self.__A2

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network.
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

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neural network.

        X is a numpy.ndarray with shape (nx, m) that contains the input data,
        where nx is the number of input features to the neuron, and m is the number of examples.
        Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data.
        iterations is the number of iterations to train over.
        alpha is the learning rate.

        Returns the evaluation of the training data after iterations of training have occurred.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")

        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")

        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)

        return self.__A2, self.cost(Y, self.__A2)

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.
        """
        m = Y.shape[1]
        cost = -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1 - A))) / m
        return cost

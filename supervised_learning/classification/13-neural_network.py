#!/usr/bin/env python3
"""This is a neural network"""

import numpy as np


class NeuralNetwork:
    """Defines a binary classification neural
    network with a single hidden layer."""

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
        term1 = Y * np.log(np.clip(A, 1e-15, 1 - 1e-15))
        term2 = (1 - Y) * np.log(np.clip(1 - A, 1e-15, 1 - 1e-15))
        cost = -(1 / m) * np.sum(term1 + term2)

        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions.

        Args:
            - X (numpy.ndarray): Input data with shape (nx, m).
            - Y (numpy.ndarray): Correct labels with shape (1, m).

        Returns:
            - Tuple of numpy.ndarray and float: Predicted labels and cost.
        """
        A, _ = self.forward_prop(X)
        predictions = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)

        return predictions, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network.

        Args:
            - X (numpy.ndarray): Input data with shape (nx, m).
            - Y (numpy.ndarray): Correct labels with shape (1, m).
            - A1 (numpy.ndarray): Output of the hidden layer.
            - A2 (numpy.ndarray): Predicted output.
            - alpha (float): Learning rate.

        Updates:
            - Private attributes __W1, __b1, __W2, and __b2.
        """
        m = Y.shape[1]

        # Backward propagation
        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.dot(self.W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        # Gradient descent update
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neural network.

        Args:
            - X (numpy.ndarray): Input data with shape (nx, m).
            - Y (numpy.ndarray): Correct labels with shape (1, m).
            - iterations (int): Number of iterations to train over.
            - alpha (float): Learning rate.

        Returns:
            - Tuple of numpy.ndarray and float: Predicted labels and cost.
        """
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')

        for i in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

        return self.evaluate(X, Y)

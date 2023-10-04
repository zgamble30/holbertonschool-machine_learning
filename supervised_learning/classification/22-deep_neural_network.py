#!/usr/bin/env python3
"""Deep Neural Network class"""

import numpy as np


class DeepNeuralNetwork:
    """Deep Neural Network class"""

    def __init__(self, nx, layers):
        """
        Initialize the DeepNeuralNetwork.

        Args:
            nx (int): Number of input features.
            layers (list): List representing the number of nodes in each layer.
        """

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            # Validate layers
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError('layers must be a list of positive integers')

            # He et al. method for initializing weights
            self.__weights["W" + str(i + 1)] = np.random.randn(
                layers[i], nx) * np.sqrt(2 / nx)
            self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))
            nx = layers[i]

    @property
    def L(self):
        """Number of layers in the neural network."""
        return self.__L

    @property
    def cache(self):
        """Intermediary values of the network."""
        return self.__cache

    @property
    def weights(self):
        """Weights and biases of the network."""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculate forward propagation of the neural network.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).

        Returns:
            tuple: Output of the neural network (AL) and the cache dictionary.
        """

        self.__cache["A0"] = X

        for i in range(1, self.__L + 1):
            # Compute weighted input
            Z = np.matmul(
                self.__weights["W" + str(i)],
                self.__cache["A" + str(i - 1)]) + self.__weights["b" + str(i)]

            # Apply sigmoid activation function
            self.__cache["A" + str(i)] = 1 / (1 + np.exp(-Z))

        return self.__cache["A" + str(self.__L)], self.__cache

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculate one pass of gradient descent on the neural network.

        Args:
            Y (numpy.ndarray): Correct labels with shape (1, m).
            cache (dict): Dictionary containing all the intermediary values of the network.
            alpha (float): Learning rate.
        """

        m = Y.shape[1]
        dA = None

        for i in reversed(range(1, self.L + 1)):
            A = cache['A' + str(i)]
            A_prev = cache['A' + str(i - 1)]
            W = self.weights['W' + str(i)]
            b = self.weights['b' + str(i)]

            if i != self.L:
                dZ = dA * (1 - np.power(A, 2))
            else:
                dZ = A - Y

            dW = (1 / m) * np.dot(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            if i - 1 > 0:
                dA = np.dot(W.T, dZ)

            self.weights['W' + str(i)] -= alpha * dW
            self.weights['b' + str(i)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the deep neural network.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): Correct labels with shape (1, m).
            iterations (int): Number of iterations to train over.
            alpha (float): Learning rate.

        Returns:
            tuple: Predictions (A) and the cost of the network.
        """

        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')

        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')

        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')

        if alpha <= 0:
            raise ValueError('alpha must be positive')

        for _ in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)

        return self.evaluate(X, Y)

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Args:
            Y (numpy.ndarray): Correct labels with shape (1, m).
            A (numpy.ndarray): Activated output with shape (1, m).

        Returns:
            float: The cost.
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): Correct labels with shape (1, m).

        Returns:
            tuple: Predictions (A) and the cost of the network.
        """
        A, _ = self.forward_prop(X)
        predictions = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return predictions, cost

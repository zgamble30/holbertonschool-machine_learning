#!/usr/bin/env python3
"""
Deep Neural Network
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Deep Neural Network
    """

    def __init__(self, nx, layers):
        """
        Initializes a deep neural network.

        Arguments:
        - nx: number of input features
        - layers: list representing the number of nodes in each layer

        Raises:
        - TypeError: if nx is not an integer or if layers is not a list of
                     positive integers
        - ValueError: if nx or any value in layers is less than 1
        """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(layers) != list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i, n in enumerate(layers):
            if type(n) != int or n < 1:
                raise TypeError("layers must be a list of positive integers")

            if i == 0:
                self.__weights["W1"] = np.random.randn(n, nx) * np.sqrt(2 / nx)
            else:
                prev_n = layers[i - 1]
                self.__weights["W" + str(i + 1)] = np.random.randn(n, prev_n) * \
                                                  np.sqrt(2 / prev_n)

            self.__weights["b" + str(i + 1)] = np.zeros((n, 1))

    @property
    def L(self):
        """
        Getter method for the number of layers in the neural network.
        """
        return self.__L

    @property
    def cache(self):
        """
        Getter method for the dictionary holding intermediate values.
        """
        return self.__cache

    @property
    def weights(self):
        """
        Getter method for the weights and biases of the neural network.
        """
        return self.__weights

    def forward_prop(self, X):
        """
        Performs forward propagation for the neural network.

        Arguments:
        - X: numpy.ndarray with shape (nx, m) that contains the input data

        Returns:
        - A: numpy.ndarray with shape (1, m) containing the activated output
             of the neural network
        - cache: dictionary holding intermediate values
        """
        self.__cache["A0"] = X

        for i in range(1, self.__L + 1):
            W = self.__weights["W" + str(i)]
            b = self.__weights["b" + str(i)]
            A_prev = self.__cache["A" + str(i - 1)]

            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))

            self.__cache["A" + str(i)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the neural network's predictions.

        Arguments:
        - Y: numpy.ndarray with shape (1, m) containing the correct labels
        - A: numpy.ndarray with shape (1, m) containing the predicted labels

        Returns:
        - cost: the cost of the neural network's predictions
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions.

        Arguments:
        - X: numpy.ndarray with shape (nx, m) that contains the input data
        - Y: numpy.ndarray with shape (1, m) containing the correct labels

        Returns:
        - A: numpy.ndarray with shape (1, m) containing the predicted labels
        - cost: the cost of the neural network's predictions
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        return np.where(A >= 0.5, 1, 0), cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Performs one pass of gradient descent on the neural network.

        Arguments:
        - Y: numpy.ndarray with shape (1, m) that contains the correct labels
        - cache: dictionary containing all intermediate values of the network
        - alpha: learning rate

        Updates:
        - self.__weights: weights and biases of the neural network
        """
        m = Y.shape[1]
        dz = cache["A" + str(self.__L)] - Y

        for i in range(self.__L, 0, -1):
            dw = (dz @ cache["A" + str(i - 1)].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            g = cache["A" + str(i - 1)] * (1 - cache["A" + str(i - 1)])
            dz = (self.__weights["W" + str(i)].T @ dz) * g

            self.__weights["W" + str(i)] -= alpha * dw
            self.__weights["b" + str(i)] -= alpha * db

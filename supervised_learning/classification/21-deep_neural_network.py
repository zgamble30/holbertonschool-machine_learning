#!/usr/bin/env python3
"""Deep Neural Network class"""

import numpy as np


class DeepNeuralNetwork:
    """Class defining a deep neural network for binary classification."""

    @staticmethod
    def initialize_weights(nx, layers):
        """
        Initializes weights using the He et al. method.

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
            Z = np.dot(self.__weights[W_key], self.__cache['A' + str(i - 1)]) \
                + self.__weights[b_key]
            activation = 1 / (1 + np.exp(-Z))
            self.__cache[A_key] = activation

        return self.__cache['A' + str(self.__L)], self.__cache

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

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network.

        Args:
            Y (numpy.ndarray): Correct labels with shape (1, m).
            cache (dict): Dictionary containing all the intermediary values of the network.
            alpha (float): Learning rate.

        Updates the private attribute __weights.

        You are allowed to use one loop.
        """
        m = Y.shape[1]
        dz_last = self.__cache['A' + str(self.__L)] - Y

        for i in range(self.__L, 0, -1):
            A_key, W_key, b_key = 'A' + str(i), 'W' + str(i), 'b' + str(i)
            A_prev_key = 'A' + str(i - 1)

            dz = np.dot(self.weights[W_key].T, dz_last) * (self.cache[A_key] * (1 - self.cache[A_key]))
            dw = np.dot(dz, self.cache[A_prev_key].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m

            self.weights[W_key] -= alpha * dw
            self.weights[b_key] -= alpha * db

            dz_last = np.dot(self.weights[W_key].T, dz_last) * (self.cache[A_prev_key] * (1 - self.cache[A_prev_key]))


"""if __name__ == "__main__":
    lib_train = np.load('../data/Binary_Train.npz')
    X_3D, Y = lib_train['X'], lib_train['Y']
    X = X_3D.reshape((X_3D.shape[0], -1)).T

    np.random.seed(0)
    deep = DeepNeuralNetwork(X.shape[0], [5, 3, 1])
    A, cache = deep.forward_prop(X)
    deep.gradient_descent(Y, cache, 0.5)
    print(deep.weights)
"""
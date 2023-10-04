#!/usr/bin/env python3
"""Deep Neural Network class"""
import numpy as np


# Define the class DeepNeuralNetwork
class DeepNeuralNetwork:
    """Deep Neural Network class"""

    # Class constructor
    def __init__(self, nx, layers):
        """
        nx is the number of input features
        layers is a list representing the number of nodes in each
        layer of the network
        """

        # Number of layers in the neural network
        self.__L = len(layers)

        # Intermediary values of the network
        self.__cache = {}

        # Weights and biases dictionary
        self.__weights = {}

        # Loop over the layers to create weights and biases for each
        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError('layers must be a list of positive integers')

            # Initialize weights using He et al. method and biases to 0
            self.__weights["W" + str(i + 1)] = np.random.randn(
                layers[i], nx) * np.sqrt(2 / nx)
            self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

            # Update nx to the number of nodes in the current layer
            nx = layers[i]

    # Getter for L
    @property
    def L(self):
        """Number of layers in the neural network"""
        return self.__L

    # Getter for cache
    @property
    def cache(self):
        """Intermediary values of the network"""
        return self.__cache

    # Getter for weights
    @property
    def weights(self):
        """Weights and biases of the network"""
        return self.__weights

    # Perform forward propagation
    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        Returns: the output of the neural network and the cache,
        respectively
        """

        # Store the input data in the cache
        self.__cache["A0"] = X

        # Loop over the layers using forward propagation
        for i in range(1, self.__L + 1):
            # Compute the weighted input (synaptic input)
            Z = np.matmul(
                self.__weights["W" + str(i)],
                self.__cache["A" + str(i - 1)]) + self.__weights["b" + str(i)]

            # Apply the sigmoid activation function
            self.__cache["A" + str(i)] = 1 / (1 + np.exp(-Z))

        # Return the output of the network and the cache
        return self.__cache["A" + str(self.__L)], self.__cache

    # Apply one pass of gradient descent
    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
        cache is a dictionary containing all the intermediary values of the network
        alpha is the learning rate
        """

        # Number of data points
        m = Y.shape[1]

        # Copy of the weights dictionary
        weights_copy = self.__weights.copy()

        # Loop over the layers in reverse order
        for i in reversed(range(self.L)):
            # If it's the last layer
            if i == self.L - 1:
                # Compute the difference between the predicted and actual values
                dz = cache['A' + str(i + 1)] - Y

            else:
                # Compute the dot product between the weights and dz
                da = np.dot(weights_copy['W' + str(i + 2)].T, dz)

                # Compute dz for the current layer
                dz = da * cache['A' + str(i + 1)] * (
                    1 - cache['A' + str(i + 1)])

            # Compute dw and db for the current layer
            dw = np.dot(dz, cache['A' + str(i)].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m

            # Update the weights and biases
            self.__weights['W' + str(i + 1)] -= alpha * dw
            self.__weights['b' + str(i + 1)] -= alpha * db

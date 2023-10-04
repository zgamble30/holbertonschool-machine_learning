#!/usr/bin/env python3
"""Deep neural network module"""

import numpy as np

class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification."""

    def __init__(self, nx, layers):
        """Class constructor"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not all(isinstance(i, int) and i > 0 for i in layers) or not layers:
            raise TypeError('layers must be a list of positive integers')

        self.__L, self.__cache, self.__weights = len(layers), {}, {}
        for l in range(self.L):
            W, b = 'W' + str(l + 1), 'b' + str(l + 1)
            if l == 0:
                self.weights[W] = np.random.randn(layers[0], nx) * np.sqrt(2 / nx)
            else:
                self.weights[W] = np.random.randn(layers[l], layers[l - 1]) * np.sqrt(2 / layers[l - 1])
            self.weights[b] = np.zeros((layers[l], 1))

    @property
    def L(self):
        """Getter method for L (number of layers)."""
        return self.__L

    @property
    def cache(self):
        """Getter method for cache (intermediary values)."""
        return self.__cache

    @property
    def weights(self):
        """Getter method for weights (weights and biases)."""
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network."""
        self.cache['A0'] = X
        for l in range(1, self.L + 1):
            W, b, A, new_A = 'W' + str(l), 'b' + str(l), 'A' + str(l - 1), 'A' + str(l)
            Z = np.dot(self.weights[W], self.cache[A]) + self.weights[b]
            self.cache[new_A] = 1 / (1 + np.exp(-Z))
        return self.cache[new_A], self.cache
#!/usr/bin/env python3
"""Deep neural network module"""

import numpy as np

class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification."""

    def __init__(self, nx, layers):
        """Class constructor"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not all(isinstance(i, int) and i > 0 for i in layers) or not layers:
            raise TypeError('layers must be a list of positive integers')

        self.__L, self.__cache, self.__weights = len(layers), {}, {}
        for l in range(self.L):
            W, b = 'W' + str(l + 1), 'b' + str(l + 1)
            if l == 0:
                self.weights[W] = np.random.randn(layers[0], nx) * np.sqrt(2 / nx)
            else:
                self.weights[W] = np.random.randn(layers[l], layers[l - 1]) * np.sqrt(2 / layers[l - 1])
            self.weights[b] = np.zeros((layers[l], 1))

    @property
    def L(self):
        """Getter method for L (number of layers)."""
        return self.__L

    @property
    def cache(self):
        """Getter method for cache (intermediary values)."""
        return self.__cache

    @property
    def weights(self):
        """Getter method for weights (weights and biases)."""
        return self.__weights

    @L.setter
    def L(self, value):
        """Setter method for L (number of layers)."""
        if type(value) is not int:
            raise TypeError('L must be an integer')
        if value < 1:
            raise ValueError('L must be a positive integer')
        self.__L = value

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network."""
        self.cache['A0'] = X
        for l in range(1, self.L + 1):
            W, b, A, new_A = 'W' + str(l), 'b' + str(l), 'A' + str(l - 1), 'A' + str(l)
            Z = np.dot(self.weights[W], self.cache[A]) + self.weights[b]
            self.cache[new_A] = 1 / (1 + np.exp(-Z))
        return self.cache[new_A], self.cache

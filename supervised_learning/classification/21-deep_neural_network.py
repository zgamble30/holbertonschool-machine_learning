import numpy as np


class DeepNeuralNetwork:
    """Deep Neural Network class"""

    def __init__(self, nx, layers):
        """Class constructor"""

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError('layers must be a list of positive integers')
            self.__weights["W" + str(i + 1)] = np.random.randn(
                layers[i], nx) * np.sqrt(2 / nx)
            self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))
            nx = layers[i]

    @property
    def L(self):
        """Getter for L"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__cache["A0"] = X
        for i in range(1, self.__L + 1):
            Z = np.matmul(
                self.__weights["W" + str(i)],
                self.__cache["A" + str(i - 1)]) + self.__weights["b" + str(i)]
            self.__cache["A" + str(i)] = 1 / (1 + np.exp(-Z))
        return self.__cache["A" + str(self.__L)], self.__cache

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        m = Y.shape[1]
        weights_copy = self.__weights.copy()
        for i in reversed(range(self.L)):
            if i == self.L - 1:
                dz = cache['A' + str(i + 1)] - Y
            else:
                da = np.dot(weights_copy['W' + str(i + 2)].T, dz)
                dz = da * cache['A' + str(i + 1)] * (
                    1 - cache['A' + str(i + 1)])
            dw = np.dot(dz, cache['A' + str(i)].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            self.__weights['W' + str(i + 1)] -= alpha * dw
            self.__weights['b' + str(i + 1)] -= alpha * db

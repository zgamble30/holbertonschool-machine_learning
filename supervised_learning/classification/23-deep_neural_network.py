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
        Calculate one pass of gradient descent on the neural network.

        Args:
            Y (numpy.ndarray): Correct labels with shape (1, m).
            cache (dict): Dictionary containing
            all the intermediary values of the network.
            alpha (float): Learning rate.
        """

        m = Y.shape[1]
        dZ = self.__cache["A" + str(self.L)] - Y

        for i in reversed(range(1, self.L + 1)):
            A_prev = self.__cache["A" + str(i - 1)]

            dW = (1 / m) * np.dot(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            if i > 1:
                dZ = np.dot(self.__weights["W" + str(i)].T, dZ) \
                    * A_prev * (1 - A_prev)

            self.__weights['W' + str(i)] -= alpha * dW
            self.__weights['b' + str(i)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the deep neural network.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): Correct labels with shape (1, m).
            iterations (int): Number of iterations to train over.
            alpha (float): Learning rate.
            verbose (bool): Whether or not to print information about the training.
            graph (bool): Whether or not to graph information about the training.
            step (int): Step for printing and graphing.

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

        if not isinstance(step, int):
            raise TypeError('step must be an integer')

        if step <= 0 or step > iterations:
            raise ValueError('step must be positive and <= iterations')

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)

            if verbose and i % step == 0:
                cost = self.cost(Y, A)
                print('Cost after {} iterations: {}'.format(i, cost))

        if graph or verbose:
            cost = self.cost(Y, A)
            print('Cost after {} iterations: {}'.format(iterations, cost))

        if graph:
            x_vals = list(range(0, iterations + 1, step))
            y_vals = [self.cost(Y, self.forward_prop(X)[0]) for _ in x_vals]
            plt.plot(x_vals, y_vals, 'b-')
            plt.xlabel('Iteration')
            plt.ylabel('Cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)


"""if __name__ == "__main__":
    import matplotlib.pyplot as plt

    lib_train = np.load('../data/Binary_Train.npz')
    X_train_3D, Y_train = lib_train['X'], lib_train['Y']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
    lib_dev = np.load('../data/Binary_Dev.npz')
    X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
    X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

    np.random.seed(0)
    deep = DeepNeuralNetwork(X_train.shape[0], [5, 3, 1])
    A, cost = deep.train(X_train, Y_train)
    accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
    print("Train cost:", cost)
    print("Train accuracy: {}%".format(accuracy))
    A, cost = deep.evaluate(X_dev, Y_dev)
    accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
    print("Dev cost:", cost)
    print("Dev accuracy: {}%".format(accuracy))
"""

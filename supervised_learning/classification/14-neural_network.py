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
            - Y (numpy.ndarray): Correct labels for
              the input data with shape (1, m).
            - A (numpy.ndarray): Activated output of
              the neuron for each example with shape (1, m).

        Returns:
            - float: The cost of the model.
        """
        m = Y.shape[1]
        cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions.

        Args:
            - X (numpy.ndarray): Input data with shape (nx, m).
            - Y (numpy.ndarray): Correct labels for
              the input data with shape (1, m).

        Returns:
            - Tuple of numpy.ndarray: The neuron’s
              prediction and the cost of the network.
        """
        _, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        prediction = np.where(A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network.

        Args:
            - X (numpy.ndarray): Input data with shape (nx, m).
            - Y (numpy.ndarray): Correct labels for
              the input data with shape (1, m).
            - A1 (numpy.ndarray): Output of the hidden layer.
            - A2 (numpy.ndarray): Predicted output.
            - alpha (float): Learning rate.

        Updates:
            - Private attributes __W1, __b1, __W2, and __b2.
        """
        m = Y.shape[1]

        dz2 = A2 - Y
        dw2 = np.dot(dz2, A1.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m

        dz1 = np.dot(self.W2.T, dz2) * (A1 * (1 - A1))
        dw1 = np.dot(dz1, X.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m

        self.__W2 = self.W2 - alpha * dw2
        self.__b2 = self.b2 - alpha * db2
        self.__W1 = self.W1 - alpha * dw1
        self.__b1 = self.b1 - alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neural network.

        Args:
            - X (numpy.ndarray): Input data with shape (nx, m).
            - Y (numpy.ndarray): Correct labels for
              the input data with shape (1, m).
            - iterations (int): Number of iterations to train over.
            - alpha (float): Learning rate.

        Returns:
            - Tuple of numpy.ndarray: The neuron’s
              prediction and the cost of the network.
        """
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')

        for i in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

        return self.evaluate(X, Y)


"""if __name__ == "__main__":
    lib_train = np.load('../data/Binary_Train.npz')
    X_3D_train, Y_train = lib_train['X'], lib_train['Y']
    X_train = X_3D_train.reshape((X_3D_train.shape[0], -1)).T

    lib_dev = np.load('../data/Binary_Dev.npz')
    X_3D_dev, Y_dev = lib_dev['X'], lib_dev['Y']
    X_dev = X_3D_dev.reshape((X_3D_dev.shape[0], -1)).T

    np.random.seed(0)
    nn = NeuralNetwork(X_train.shape[0], 3)
    A, cost = nn.train(X_train, Y_train, iterations=100)
    accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
    print("Train cost:", cost)
    print("Train accuracy: {}%".format(accuracy))

    A, cost = nn.evaluate(X_dev, Y_dev)
    accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
    print("Dev cost:", cost)
    print("Dev accuracy: {}%".format(accuracy))
"""

#!/usr/bin/env python3

import numpy as np


class NeuralNetwork:
    def __init__(self, nx, nodes):
        """
        Initialize the neural network.

        Arguments:
        - nx: Number of input features.
        - nodes: Number of nodes in the hidden layer.
        """
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    def sigmoid(self, Z):
        """
        Compute the sigmoid activation function.

        Arguments:
        - Z: Input value(s).

        Returns:
        - The sigmoid activation of Z.
        """
        return 1 / (1 + np.exp(-Z))

    def forward_prop(self, X):
        """
        Perform forward propagation.

        Arguments:
        - X: Input data.

        Returns:
        - A tuple containing the activations of the hidden and output layers.
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = self.sigmoid(Z1)
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = self.sigmoid(Z2)
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Compute the cost function.

        Arguments:
        - Y: Correct labels.
        - A: Predicted labels.

        Returns:
        - The cost value.
        """
        m = Y.shape[1]
        cost = (-1 / m) * (
            np.matmul(Y, np.log(A).T) +
            np.matmul(1 - Y, np.log(1.0000001 - A).T)
        )
        return cost[0, 0]

    def evaluate(self, X, Y):
        """
        Evaluate the neural network's predictions.

        Arguments:
        - X: Input data.
        - Y: Correct labels.

        Returns:
        - A tuple containing the predicted labels and the cost.
        """
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        predictions = np.where(A2 >= 0.5, 1, 0)
        return predictions, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Perform gradient descent to update weights and biases.

        Arguments:
        - X: Input data.
        - Y: Correct labels.
        - A1: Hidden layer activations.
        - A2: Output layer activations.
        - alpha: Learning rate.

        Returns:
        - None
        """
        m = Y.shape[1]
        dZ2 = A2 - Y
        dW2 = (1 / m) * np.matmul(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.matmul(self.__W2.T, dZ2)
        dZ1 *= A1 * (1 - A1)
        dW1 = (1 / m) * np.matmul(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Train the neural network.

        Arguments:
        - X: Input data.
        - Y: Correct labels.
        - iterations: Number of iterations to train over.
        - alpha: Learning rate.

        Returns:
        - The evaluation of the training data after training.
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
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

        return self.evaluate(X, Y)

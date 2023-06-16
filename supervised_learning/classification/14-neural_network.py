#!/usr/bin/env python3

import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size):
        """
        Initializes the parameters of the neural network
        """

        np.random.seed(0)
        self.W1 = np.random.randn(hidden_size, input_size)
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(1, hidden_size)
        self.b2 = 0

    def sigmoid(self, Z):
        """
        Applies the sigmoid activation function element-wise
        """

        return 1 / (1 + np.exp(-Z))

    def forward_prop(self, X):
        """
        Performs forward propagation to calculate the outputs
        """

        Z1 = np.dot(self.W1, X) + self.b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = self.sigmoid(Z2)

        return A1, A2

    def backward_prop(self, X, Y, A1, A2):
        """
        Performs backward propagation to update the parameters
        """

        m = Y.shape[1]

        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.dot(self.W2.T, dZ2) * (1 - np.power(A1, 2))
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        return dW1, db1, dW2, db2

    def train(self, X, Y, iterations=1000, alpha=0.01):
        """
        Trains the neural network on the given data
        """

        for i in range(iterations):
            A1, A2 = self.forward_prop(X)
            dW1, db1, dW2, db2 = self.backward_prop(X, Y, A1, A2)

            self.W1 -= alpha * dW1
            self.b1 -= alpha * db1
            self.W2 -= alpha * dW2
            self.b2 -= alpha * db2

        predictions, cost, accuracy = self.evaluate(X, Y)

        return predictions, cost, accuracy

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions
        """

        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        predictions = np.round(A2)
        accuracy = np.mean(predictions == Y) * 100

        return predictions, cost, accuracy

    def cost(self, Y, A):
        """
        Calculates the cross-entropy cost
        """

        m = Y.shape[1]
        cost = -1 / m * (np.dot(Y, np.log(A).T) + np.dot(1 - Y, np.log(1 - A).T))

        return np.squeeze(cost)

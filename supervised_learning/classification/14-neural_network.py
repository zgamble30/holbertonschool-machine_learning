#!/usr/bin/env python3

"""
Neural Network Class
"""

import numpy as np


class NeuralNetwork:
    """Neural Network class"""

    def __init__(self, nx, nodes):
        """
        Initialize the NeuralNetwork object

        Arguments:
          - nx (int): number of input features
          - nodes (int): number of nodes in the hidden layer

        Attributes:
          - W1 (numpy.ndarray): weights vector for the hidden layer
          - b1 (numpy.ndarray): bias for the hidden layer
          - A1 (float): activated output for the hidden layer
          - W2 (numpy.ndarray): weights vector for the output neuron
          - b2 (float): bias for the output neuron
          - A2 (float): activated output for the output neuron
        """

        # Initialize weights using He et al. method
        self.W1 = np.random.randn(nodes, nx) * np.sqrt(2 / nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.randn(1, nodes) * np.sqrt(2 / nodes)
        self.b2 = 0
        self.A2 = 0

    def forward_prop(self, X):
        """
        Perform forward propagation

        Arguments:
          - X (numpy.ndarray): input data (shape: m x nx)

        Returns:
          The activated output (self.A1, self.A2)
        """

        # Calculate hidden layer's activated output
        Z1 = np.matmul(self.W1, X.T) + self.b1
        self.A1 = 1 / (1 + np.exp(-Z1))

        # Calculate output neuron's activated output
        Z2 = np.matmul(self.W2, self.A1) + self.b2
        self.A2 = 1 / (1 + np.exp(-Z2))

        return self.A1, self.A2

    def cost(self, Y, A):
        """
        Calculate the cost of the model using logistic regression

        Arguments:
          - Y (numpy.ndarray): correct labels (shape: 1 x m)
          - A (numpy.ndarray): predicted labels (shape: 1 x m)

        Returns:
          The cost
        """

        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate the neural network's predictions

        Arguments:
          - X (numpy.ndarray): input data (shape: m x nx)
          - Y (numpy.ndarray): correct labels (shape: 1 x m)

        Returns:
          The neuron's prediction and the cost
        """

        _, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        prediction = np.where(A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Perform one pass of gradient descent on the neural network

        Arguments:
          - X (numpy.ndarray): input data (shape: m x nx)
          - Y (numpy.ndarray): correct labels (shape: 1 x m)
          - A1 (numpy.ndarray): activated output of the hidden layer
          - A2 (numpy.ndarray): activated output of the output neuron
          - alpha (float): learning rate

        Updates:
          Updates the neural network's weights and biases
        """

        m = Y.shape[1]

        # Backpropagation for output layer
        dZ2 = A2 - Y
        dW2 = np.matmul(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        # Backpropagation for hidden layer
        dZ1 = np.matmul(self.W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = np.matmul(dZ1, X) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        # Update weights and biases
        self.W1 -= alpha * dW1
        self.b1 -= alpha * db1
        self.W2 -= alpha * dW2
        self.b2 -= alpha * db2

#!/usr/bin/env python3
"""Neural Network with one hidden layer for binary classification"""

import numpy as np


class NeuralNetwork:
    """
    Defines a neural network with one hidden layer for binary classification.
    """

    def __init__(self, nx, nodes):
        """
        Initializes the neural network.

        Args:
            - nx (int): Number of input features.
            - nodes (int): Number of nodes in the hidden layer.

        Raises:
            - TypeError: If nx or nodes is not an integer.
            - ValueError: If nx or nodes is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initialize weights and biases for the hidden layer
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros(shape=(nodes, 1))
        self.A1 = 0

        # Initialize weights and biases for the output neuron
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0

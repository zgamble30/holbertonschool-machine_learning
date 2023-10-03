#!/usr/bin/env python3
"""Neural Network"""

import numpy as np


class NeuralNetwork:
    """Defines a neural network with one hidden layer
    performing binary classification"""

    def __init__(self, nx, nodes):
        """Class constructor"""
        self.nx = nx
        self.nodes = nodes

        self.initialize_weights()

    def initialize_weights(self):
        """Initializes weights and biases for the neural network"""
        # Validate input types and values
        if not isinstance(self.nx, int):
            raise TypeError('nx must be an integer')
        if self.nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(self.nodes, int):
            raise TypeError('nodes must be an integer')
        if self.nodes < 1:
            raise ValueError('nodes must be a positive integer')

        # Initialize weights and biases for the hidden layer
        self.__W1 = np.random.randn(self.nodes, self.nx)
        self.__b1 = np.zeros((self.nodes, 1))
        self.__A1 = 0

        # Initialize weights and biases for the output neuron
        self.__W2 = np.random.randn(1, self.nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for W1"""
        return self.__W1

    @property
    def b1(self):
        """Getter for b1"""
        return self.__b1

    @property
    def A1(self):
        """Getter for A1"""
        return self.__A1

    @property
    def W2(self):
        """Getter for W2"""
        return self.__W2

    @property
    def b2(self):
        """Getter for b2"""
        return self.__b2

    @property
    def A2(self):
        """Getter for A2"""
        return self.__A2

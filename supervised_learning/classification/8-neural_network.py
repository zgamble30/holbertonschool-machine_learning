#!/usr/bin/env python3
"""creating nodes in a network.... networking....a neural network"""
import numpy as np


class NeuralNetwork:
    """
    Defines a neural network with one hidden layer performing binary
    classification.
    """

    def __init__(self, nx, nodes):
        """
        Class constructor to initialize a neural network.

        nx: the number of input features to the neural network
        nodes: the number of nodes found in the hidden layer

        W1: The weights vector for the hidden layer. It is initialized using
            a random normal distribution.
        b1: The bias for the hidden layer. Initialized with 0’s.
        A1: The activated output for the hidden layer. Initialized to 0.

        W2: The weights vector for the output neuron. It is initialized using
            a random normal distribution.
        b2: The bias for the output neuron. Initialized to 0.
        A2: The activated output for the output neuron (prediction).
            Initialized to 0.
        """

        # Check if nx and nodes are of valid types and values
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initialize weights and biases for hidden layer and output neuron
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros(shape=(nodes, 1))
        self.A1 = 0
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0

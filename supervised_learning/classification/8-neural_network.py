#!/usr/bin/env python3
#Neural Network

import numpy as np

class NeuralNetwork:
    #Defines a neural network with one hidden layer

    def __init__(self, nx, nodes):
        
        """
        Class constructor to initialize the neural network.

        Args:
            nx (int): Number of input features.
            nodes (int): Number of nodes in the hidden layer.

        Raises:
            TypeError: If nx or nodes is not an integer.
            ValueError: If nx or nodes is less than 1.
        """
        # Check if nx is an integer and positive
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        # Check if nodes is an integer and positive
        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        # Initialize weights and biases for the hidden layer
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
        # Initialize weights and biases for the output neuron

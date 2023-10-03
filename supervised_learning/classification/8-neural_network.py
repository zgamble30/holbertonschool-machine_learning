#!/usr/bin/env python3
"""Neural network with one hidden layer performing binary classification"""

import numpy as np


class NeuralNetwork:
    """Defines a neural network with one hidden layer performing binary classification"""

    def __init__(self, nx, nodes):
        """
        Initializes the NeuralNetwork object.

        Parameters:
        - nx: The number of input features.
        - nodes: The number of nodes found in the hidden layer.

        Raises:
        - TypeError: If nx is not an integer or if nodes is not an integer.
        - ValueError: If nx is less than 1 or if nodes is less than 1.
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
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0

        # Initialize weights and biases for the output neuron
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0

    @property
    def W1(self):
        """Getter method for W1"""
        return self.__W1

    @property
    def b1(self):
        """Getter method for b1"""
        return self.__b1

    @property
    def A1(self):
        """Getter method for A1"""
        return self.__A1

    @property
    def W2(self):
        """Getter method for W2"""
        return self.__W2

    @property
    def b2(self):
        """Getter method for b2"""
        return self.__b2

    @property
    def A2(self):
        """Getter method for A2"""
        return self.__A2

    @W1.setter
    def W1(self, value):
        """Setter method for W1"""
        self.__W1 = value

    @b1.setter
    def b1(self, value):
        """Setter method for b1"""
        self.__b1 = value

    @A1.setter
    def A1(self, value):
        """Setter method for A1"""
        self.__A1 = value

    @W2.setter
    def W2(self, value):
        """Setter method for W2"""
        self.__W2 = value

    @b2.setter
    def b2(self, value):
        """Setter method for b2"""
        self.__b2 = value

    @A2.setter
    def A2(self, value):
        """Setter method for A2"""
        self.__A2 = value


if __name__ == "__main__":
    import numpy as np

    NN = NeuralNetwork

    lib_train = np.load('../data/Binary_Train.npz')
    X_3D, Y = lib_train['X'], lib_train['Y']
    X = X_3D.reshape((X_3D.shape[0], -1)).T

    np.random.seed(0)
    nn = NN(X.shape[0], 3)
    print(nn.W1)
    print(nn.W1.shape)
    print(nn.b1)
    print(nn.W2)
    print(nn.W2.shape)
    print(nn.b2)
    print(nn.A1)
    print(nn.A2)
    nn.A1 = 10
    print(nn.A1)

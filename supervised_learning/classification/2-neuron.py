#!/usr/bin/env python3
"""My third attempt at creating a neuron"""
import numpy as np


class Neuron:
    """Just a solo neuron for binary classification"""

    def __init__(self, nx):
        """
        Initialize a neuron with private attributes.

        Parameters:
        - nx: The number of input features to the neuron.

        Raises:
        - TypeError: If nx is not an integer.
        - ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter method for __W"""
        return self.__W

    @property
    def b(self):
        """Getter method for __b"""
        return self.__b

    @property
    def A(self):
        """Getter method for __A"""
        return self.__A

    def forward_prop(self, X):
        """
        calculates the forward propagation of the neuron.

        Parameters:
        - X: numpy.ndarray with shape (nx, m) containing the input data.
             nx is the number of input features to the neuron.
             m is the number of examples.

        Returns:
        - The private attribute __A.
        """
        z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

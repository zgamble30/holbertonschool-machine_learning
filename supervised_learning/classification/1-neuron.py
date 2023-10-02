#!/usr/bin/env python3
"""My second attempt at creating a neuron"""
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

    def get_weights(self):
        """Getter method for __W"""
        return self.__W

    def get_bias(self):
        """Getter method for __b"""
        return self.__b

    def get_activation(self):
        """Getter method for __A"""
        return self.__A

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
            w - weights vector
            b - biases for the neuron
            a - activated output or the prediction
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
        """this is the getter for W"""
        return self.__W

    @property
    def b(self):
        """this is the getter for b"""
        return self.__b

    @property
    def A(self):
        """this is the getter for A"""
        return self.__A

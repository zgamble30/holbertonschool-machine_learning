#!/usr/bin/env python3
""" neural network class"""
import numpy as np


class NeuralNetwork:
    """ Creating Neural Network class"""

    def __init__(self, nx, nodes):
        """ initialize class"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) != int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ return private w1"""
        return self.__W1

    @property
    def b1(self):
        """ return private b1"""
        return self.__b1

    @property
    def A1(self):
        """ return private A1"""
        return self.__A1

    @property
    def W2(self):
        """ return private w2"""
        return self.__W2

    @property
    def b2(self):
        """ return private b2"""
        return self.__b2

    @property
    def A2(self):
        """ return private a2"""
        return self.__A2

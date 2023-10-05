#!/usr/bin/env python3
"""This is a neural network"""

import numpy as np

class NeuralNetwork:
    def __init__(self, n_x, n_h, n_y=1):
        self.W1 = np.random.randn(n_x, n_h)
        self.b1 = np.zeros((1, n_h))
        self.W2 = np.random.randn(n_h, n_y)
        self.b2 = np.zeros((1, n_y))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_prop(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A1, self.A2

    def cost(self, Y, A):
        m = Y.shape[1]
        cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

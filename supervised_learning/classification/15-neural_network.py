#!/usr/bin/env python3
"""Neural Network"""
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork():
    """
    Neural Network
    """

    def __init__(self, nx, nodes):
        """
        - Initialize a neural network
        - nx is the number of input features
        - nodes is the number of nodes found in the hidden layer
        """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(nodes) != int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.normal(0, 1, (nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(0, 1, (1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ Get the W1 weight
        Returns:
            list of lists of weigths
        """
        return self.__W1

    @property
    def b1(self):
        """ Get the b1 bias
        Returns:
            bias
        """
        return self.__b1

    @property
    def A1(self):
        """ Get the A1 activation function
        Returns:
        """
        return self.__A1

    @property
    def W2(self):
        """ Get the W2 weight
        Returns:
            list of lists of weigths
        """
        return self.__W2

    @property
    def b2(self):
        """ Get the b2 bias
        Returns:
            bias
        """
        return self.__b2

    @property
    def A2(self):
        """ Get the A2 activation function
        Returns:
        """
        return self.__A2

    def forward_prop(self, X):
        """
        - Calculates the forward propagation of the neural network
        - X is a numpy.ndarray with shape (nx, m) that contains the input data,
        nx is the number of input features to the neuron and m is the number
        of examples.
        """
        # W1 (nodes, nx), X (nx, m), b1 (nodes, 1)
        z1 = self.__W1 @ X + self.__b1
        self.__A1 = 1 / (1 + np.exp(-1 * z1))
        # W2 (1, nodes), A1 (nodes, m), b2 (1, 1)
        z2 = self.__W2 @ self.__A1 + self.__b2
        self.__A2 = 1 / (1 + np.exp(-1 * z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        - Calculates the cost of the model using logistic regression
        - Y is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data.
        - A is a numpy.ndarray with shape (1, m) containing the activated
        output of the neuron for each example.
        - Returns the cost
        """
        # A(1, m), Y (1, m)
        cost = -1 * np.sum(((Y * np.log(A)) + ((1 - Y) * np.log(
                1.0000001 - A))))
        cost = cost / Y.shape[1]
        return cost

    def evaluate(self, X, Y):
        """
        - Evaluates the neural network’s predictions
        - X is a numpy.ndarray with shape (nx, m) that contains the input data,
        nx is the number of input features to the neuron and m is the number
        of examples.
        - Y is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data.
        """
        _, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        A = np.where(A2 >= 0.5, 1, 0)
        return A, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        - Calculates one pass of gradient descent on the neural network
        - X is a numpy.ndarray with shape (nx, m) that contains the input data,
        nx is the number of input features to the neuron and m is the number
        of examples.
        - Y is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data.
        - A1 is the output of the hidden layer.
        - A2 is the predicted output.
        - alpha is the learning rate.
        Updates the private attributes __W1, __b1, __W2, and __b2
        """
        m = X.shape[1]
        # A2 (1, m) --> dz2 (1, m)
        dz2 = A2 - Y
        # dz2 (1, m), A1.T (m, nodes)
        dw2 = (dz2 @ A1.T) / m
        # db2 (1, 1)
        db2 = np.sum(dz2, axis=1, keepdims=True) / m
        # A1 (nodes, m) --> g1 (nodes, m)
        g1 = A1 * (1 - A1)
        # __W2 (nodes, 1), dz2 (1, m) --> (nodes, m) * g1(nodes, m)
        # --> dz1 (nodes, m)
        dz1 = (self.__W2.T @ dz2) * g1
        # dz1(nodes, m), X.T (m, nx) --> dw1 (nodes, nx)
        dw1 = (dz1 @ X.T) / m
        # db1 (nodes, 1)
        db1 = np.sum(dz1, axis=1, keepdims=True) / m
        # __W1 (nodes, nx), __b1 (nodes, 1)
        self.__W1 = self.__W1 - (alpha * dw1)
        self.__b1 = self.__b1 - (alpha * db1)
        # __W2 (1, nodes), __b2 (1, 1)
        self.__W2 = self.__W2 - (alpha * dw2)
        self.__b2 = self.__b2 - (alpha * db2)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        - Trains the neural network.
        - X is a numpy.ndarray with shape (nx, m) that contains the input data,
        nx is the number of input features to the neuron and m is the number
        of examples.
        - Y is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data.
        - iterations is the number of iterations to train over
        """
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha < 0.0:
            raise ValueError("alpha must be positive")
        # check if 0 <= step <= iterations
        if verbose and graph:
            if type(step) != int:
                raise TypeError("step must be an integer")
            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        cst = []
        it = []
        for epoc in range(iterations):
            self.__A1, self.__A2 = self.forward_prop(X)
            # alpha = 0.5 give better results than 0.05
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
            c = self.cost(Y, self.__A2)
            cst.append(c)
            it.append(epoc)
            if verbose and epoc % step == 0:
                print("Cost after {} iterations: {}".format(epoc, c))
        if verbose and (epoc + 1) % step == 0:
            print("Cost after {} iterations: {}".format(epoc + 1, c))
        if graph:
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.plot(it, cst, 'b-')
            plt.show()
        A, c = self.evaluate(X, Y)
        return A, c

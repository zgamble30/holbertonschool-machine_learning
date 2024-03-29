#!/usr/bin/env python3
"""This is a shebang"""
import numpy as np
import matplotlib.pyplot as plt


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
        Calculate the forward propagation of the neuron.

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

    def cost(self, Y, A):
        """
        Calculate the cost of the model using logistic regression.

        Parameters:
        - Y: numpy.ndarray with shape (1, m) containing the correct labels.
        - A: numpy.ndarray with shape (1, m) containing the activated output.

        Returns:
        - The cost.
        """
        m = Y.shape[1]
        cost = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate the neuron's predictions.

        Parameters:
        - X: numpy.ndarray with shape (nx, m) containing the input data.
             nx is the number of input features to the neuron.
             m is the number of examples.
        - Y: numpy.ndarray with shape (1, m) containing the correct labels.

        Returns:
        - A tuple containing the neuron's prediction (numpy.ndarray with shape (1, m))
          and the cost of the network (float).
        """
        A = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Perform one pass of gradient descent on the neuron.

        Parameters:
        - X: numpy.ndarray with shape (nx, m) that contains the input data.
             nx is the number of input features to the neuron.
             m is the number of examples.
        - Y: numpy.ndarray with shape (1, m) containing the correct labels for the input data.
        - A: numpy.ndarray with shape (1, m) containing the activated output of the neuron for each example.
        - alpha: The learning rate.

        Updates:
        - The private attributes __W and __b.
        """
        m = Y.shape[1]

        dz = A - Y
        dw = np.dot(dz, X.T) / m
        db = np.sum(dz) / m

        self.__W -= alpha * dw
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Train the neuron with additional features for verbose printing and graph plotting.

        Parameters:
        - X: numpy.ndarray with shape (nx, m) that contains the input data.
             nx is the number of input features to the neuron.
             m is the number of examples.
        - Y: numpy.ndarray with shape (1, m) containing the correct labels for the input data.
        - iterations: The number of iterations to train over.
        - alpha: The learning rate.
        - verbose: A boolean that defines whether or not to print information about the training.
        - graph: A boolean that defines whether or not to graph information about the training.
        - step: The step for printing and graphing.

        Raises:
        - TypeError: If iterations is not an integer.
        - ValueError: If iterations is not positive.
        - TypeError: If alpha is not a float.
        - ValueError: If alpha is not positive.
        - TypeError: If step is not an integer.
        - ValueError: If step is not positive or is greater than iterations.

        Returns:
        - The evaluation of the training data after iterations of training have occurred.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step <= 0 or step > iterations:
            raise ValueError("step must be positive and <= iterations")

        costs = []
        for iteration in range(iterations + 1):
            A = self.forward_prop(X)
            cost = self.cost(Y, A)
            self.gradient_descent(X, Y, A, alpha)

            if verbose and iteration % step == 0:
                print(f"Cost after {iteration} iterations: {cost}")

            costs.append(cost)

        if graph:
            x_vals = [i for i in range(0, iterations + 1, step)]
            plt.plot(x_vals, costs, 'b-')
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

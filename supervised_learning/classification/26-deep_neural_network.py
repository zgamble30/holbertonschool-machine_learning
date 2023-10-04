#!/usr/bin/env python3
"""Deep Neural Network Module"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """Class representing a deep neural network for binary classification."""

    def __init__(self, input_size, layer_sizes):
        """
        Initialize the DeepNeuralNetwork.

        Args:
            input_size (int): Number of input features.
            layer_sizes (list): List representing the number of nodes in each layer.
        """
        self._validate_input(input_size, layer_sizes)
        self._initialize_weights(input_size, layer_sizes)

    def _validate_input(self, input_size, layer_sizes):
        """Validate input parameters."""
        if not isinstance(input_size, int) or input_size < 1:
            raise ValueError("Input size must be a positive integer.")
        if not isinstance(layer_sizes, list) or len(layer_sizes) < 1 or any(size < 1 for size in layer_sizes):
            raise ValueError("Layer sizes must be a list of positive integers.")

    def _initialize_weights(self, input_size, layer_sizes):
        """Initialize weights and biases using He et al. method."""
        self.__L = len(layer_sizes)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            prev_size = layer_sizes[i - 1] if i > 0 else input_size
            weight_scale = np.sqrt(2 / prev_size)
            weights = np.random.randn(layer_sizes[i], prev_size) * weight_scale

            self.__weights[f'W{i + 1}'] = weights
            self.__weights[f'b{i + 1}'] = np.zeros((layer_sizes[i], 1))

    def _sigmoid(self, X):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-X))

    def _forward_propagation(self, X):
        """Perform forward propagation."""
        A = X
        self.__cache['A0'] = X

        for i in range(1, self.__L + 1):
            W = self.__weights[f'W{i}']
            b = self.__weights[f'b{i}']
            Z = np.matmul(W, A) + b
            A = self._sigmoid(Z)
            self.__cache[f'A{i}'] = A

        return A, self.__cache

    def _cost(self, Y, A):
        """Compute the cross-entropy cost."""
        m = Y.shape[1]
        epsilon = 1e-10  # Small value to avoid log(0) issues
        cost = -1 / m * np.sum(Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon))
        return cost

    def _backward_propagation(self, Y):
        """Perform backward propagation and update weights."""
        m = Y.shape[1]
        dZ = self.__cache[f'A{self.__L}'] - Y

        for i in range(self.__L, 0, -1):
            A_prev = self.__cache[f'A{i - 1}']
            W = self.__weights[f'W{i}']
            b = self.__weights[f'b{i}']

            dW = 1 / m * np.matmul(dZ, A_prev.T)
            db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            dA = np.matmul(W.T, dZ)

            self.__weights[f'W{i}'] -= self.__alpha * dW
            self.__weights[f'b{i}'] -= self.__alpha * db

            if i > 1:
                dZ = dA * (A_prev * (1 - A_prev))

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Train the neural network.

        Args:
            X (numpy.ndarray): Input data with shape (input_size, m).
            Y (numpy.ndarray): True labels with shape (1, m).
            iterations (int): Number of training iterations.
            alpha (float): Learning rate.
            verbose (bool): If True, print cost after each iteration.
            graph (bool): If True, plot the training cost.
            step (int): Print and plot cost every 'step' iterations.

        Returns:
            tuple: Tuple containing predictions and final cost after training.
        """
        self.__alpha = alpha

        X_grp, Y_grp = [], []

        for i in range(iterations):
            A, cache = self._forward_propagation(X)
            self._backward_propagation(Y)

            if verbose and (i == 0 or i % step == 0):
                print(f"Cost after {i} iterations: {self._cost(Y, A)}")

            if graph and (i == 0 or i % step == 0):
                current_cost = self._cost(Y, A)
                Y_grp.append(current_cost)
                X_grp.append(i)

        if graph:
            plt.plot(X_grp, Y_grp)
            plt.xlabel('Iteration')
            plt.ylabel('Cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def evaluate(self, X, Y):
        """
        Evaluate the neural network on given data.

        Args:
            X (numpy.ndarray): Input data with shape (input_size, m).
            Y (numpy.ndarray): True labels with shape (1, m).

        Returns:
            tuple: Tuple containing predictions and cost on evaluation data.
        """
        A, _ = self._forward_propagation(X)
        predictions = np.where(A >= 0.5, 1, 0)

        return predictions, self._cost(Y, A)

    def save(self, filename):
        """
        Save the neural network to a file.

        Args:
            filename (str): Name of the file to save the model.
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """
        Load a pickled DeepNeuralNetwork object from a file.

        Args:
            filename (str): Name of the file to load the model from.

        Returns:
            DeepNeuralNetwork or None: Loaded object or None if the file doesn't exist.
        """
        try:
            if not filename.endswith('.pkl'):
                filename += '.pkl'

            with open(filename, 'rb') as file:
                return pickle.load(file)
        except Exception:
            return None

    @property
    def L(self):
        """Number of layers in the neural network."""
        return self.__L

    @property
    def cache(self):
        """Intermediate values during forward propagation."""
        return self.__cache

    @property
    def weights(self):
        """Weights and biases of the neural network."""
        return self.__weights

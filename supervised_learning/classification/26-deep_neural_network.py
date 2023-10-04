#!/usr/bin/env python3
"""Implementation of a Deep Neural Network."""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class CustomNeuralNetwork:
    """Custom Neural Network Class"""
    def __init__(self, input_size, hidden_layers):
        """Initialize the neural network with specified layers."""
        if not isinstance(input_size, int):
            raise TypeError("Input size must be an integer")
        if input_size < 1:
            raise ValueError("Input size must be a positive integer")
        if not isinstance(hidden_layers, list):
            raise TypeError("Hidden layers must be a list of positive integers")
        if len(hidden_layers) < 1 or False in (np.array(hidden_layers) > 0):
            raise TypeError("Hidden layers must be a list of positive integers")

        self.__num_layers = len(hidden_layers)
        self.__cache = {}
        self.__weights = {}

        for layer_idx in range(self.__num_layers):
            if layer_idx == 0:
                weights = np.random.randn(hidden_layers[layer_idx], input_size) * np.sqrt(2 / input_size)
                self.__weights[f'W{layer_idx + 1}'] = weights
            else:
                sqrt_val = np.sqrt(2 / hidden_layers[layer_idx - 1])
                weights = np.random.randn(hidden_layers[layer_idx], hidden_layers[layer_idx - 1]) * sqrt_val
                self.__weights[f'W{layer_idx + 1}'] = weights
            self.__weights[f'b{layer_idx + 1}'] = np.zeros((hidden_layers[layer_idx], 1))

    def forward_propagation(self, X):
        """Perform forward propagation through the neural network."""
        activation = X
        self.__cache['A0'] = X

        for layer_idx in range(1, self.__num_layers + 1):
            weight_matrix = self.__weights[f'W{layer_idx}']
            bias_vector = self.__weights[f'b{layer_idx}']
            linear_output = np.matmul(weight_matrix, activation) + bias_vector
            activation = self.sigmoid(linear_output)
            self.__cache[f'A{layer_idx}'] = activation

        return activation, self.__cache

    def sigmoid(self, X):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-X))

    def compute_cost(self, Y, predicted_output):
        """Compute the cost (loss) between predicted output and true labels."""
        m = Y.shape[1]
        epsilon = 1e-10
        loss = - (1 / m) * np.sum(Y * np.log(predicted_output + epsilon) + (1 - Y) * np.log(1 - predicted_output + epsilon))
        return loss

    def evaluate(self, X, Y):
        """Evaluate the performance of the neural network."""
        predicted_output, _ = self.forward_propagation(X)
        predictions = np.where(predicted_output >= 0.5, 1, 0)
        return predictions, self.compute_cost(Y, predicted_output)

    def gradient_descent(self, Y, cache, learning_rate=0.05):
        """Perform gradient descent to update weights and biases."""
        m = Y.shape[1]
        layer_idx = self.__num_layers

        current_activation = cache[f"A{layer_idx}"]
        dz = current_activation - Y

        for current_layer in range(layer_idx, 0, -1):
            prev_activation = cache[f"A{current_layer - 1}"]
            weight_matrix = self.__weights[f'W{current_layer}']
            bias_vector = self.__weights[f'b{current_layer}']

            dw = (1 / m) * np.matmul(dz, prev_activation.T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
            dz = np.matmul(weight_matrix.T, dz)

            self.__weights[f'W{current_layer}'] -= learning_rate * dw
            self.__weights[f'b{current_layer}'] -= learning_rate * db

            if current_layer > 1:
                dz *= (prev_activation * (1 - prev_activation))

    def train(self, X, Y, iterations=5000, learning_rate=0.05,
              verbose=True, plot_graph=True, plot_step=100):
        """Train the neural network using gradient descent."""
        if type(iterations) is not int:
            raise TypeError("Iterations must be an integer")
        if iterations <= 0:
            raise ValueError("Iterations must be a positive integer")
        if type(learning_rate) is not float:
            raise TypeError("Learning rate must be a float")
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        x_values = []
        y_values = []

        for epoch in range(iterations):
            predicted_output, cache = self.forward_propagation(X)
            self.gradient_descent(Y, self.__cache, learning_rate)

            if verbose:
                if epoch == 0 or epoch % plot_step == 0:
                    print("Cost after {} iterations: {}"
                          .format(epoch, self.compute_cost(Y, predicted_output)))

            if plot_graph:
                if epoch == 0 or epoch % plot_step == 0:
                    current_cost = self.compute_cost(Y, predicted_output)
                    y_values.append(current_cost)
                    x_values.append(epoch)
                plt.plot(x_values, y_values)
                plt.xlabel('Iteration')
                plt.ylabel('Cost')
                plt.title('Training Cost')

            if verbose or plot_graph:
                if type(plot_step) is not int:
                    raise TypeError("Plot step must be an integer")
                if plot_step <= 0 or plot_step > iterations:
                    raise ValueError("Plot step must be positive and <= iterations")

        if plot_graph:
            plt.show()

        return self.evaluate(X, Y)

    def save_model(self, filename):
        """Save the trained model to a file using pickle."""
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_model(filename):
        """Load a trained model from a file."""
        try:
            if not filename.endswith('.pkl'):
                filename += '.pkl'

            with open(filename, 'rb') as file:
                return pickle.load(file)
        except Exception:
            return None

    @property
    def num_layers(self):
        """Getter for the number of layers in the neural network."""
        return self.__num_layers

    @property
    def cache_data(self):
        """Getter for intermediate value cache."""
        return self.__cache

    @property
    def model_parameters(self):
        """Getter for the neural network's weights and biases."""
        return self.__weights

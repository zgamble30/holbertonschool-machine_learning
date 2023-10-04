import matplotlib.pyplot as plt
import numpy as np

class DeepNeuralNetwork:
    # ... (previous code remains unchanged)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the deep neural network.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): Correct labels with shape (1, m).
            iterations (int): Number of iterations to train over.
            alpha (float): Learning rate.
            verbose (bool): Whether or not to print training information.
            graph (bool): Whether or not to plot the training cost graph.
            step (int): Step size for printing and plotting.

        Returns:
            tuple: Predictions (A) and the cost of the network.
        """

        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')

        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')

        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')

        if alpha <= 0:
            raise ValueError('alpha must be positive')

        if not isinstance(step, int):
            raise TypeError('step must be an integer')

        if step <= 0 or step > iterations:
            raise ValueError('step must be positive and <= iterations')

        costs = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)
            
            # Print cost every step iterations
            if verbose and i % step == 0:
                print(f'Cost after {i} iterations: {cost}')

            # Store cost for plotting
            if graph and i % step == 0:
                costs.append(cost)

            self.gradient_descent(Y, cache, alpha)

        # Plotting
        if graph:
            plt.plot(range(0, iterations + 1, step), costs, marker='o')
            plt.xlabel('Iteration')
            plt.ylabel('Cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

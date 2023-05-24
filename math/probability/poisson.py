#!/usr/bin/env python3

class Poisson:
    """
    Class representing a Poisson distribution.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes a Poisson distribution.

        Args:
            data (list): List of data points to estimate the distribution.
            lambtha (float): Expected number of occurrences in a given time frame.

        Raises:
            ValueError: If lambtha is not a positive value or equals 0.
            TypeError: If data is not a list.
            ValueError: If data does not contain at least two data points.
        """

        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    @staticmethod
    def factorial(n):
        """
        Computes the factorial of a number.

        Args:
            n (int): Number to compute the factorial of.

        Returns:
            int: Factorial of the number.
        """

        if n == 0 or n == 1:
            return 1
        else:
            return n * Poisson.factorial(n - 1)

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of successes.

        Args:
            k (int): Number of successes.

        Returns:
            float: Probability mass function (PMF) value.
        """
        if not isinstance(k, int) or k < 0:
            return 0

        e_approx = 2.7182818285  # Approximation for the value of e
        lambtha = self.lambtha
        k = int(k)

        pmf_value = (lambtha ** k) * (e_approx ** (-lambtha))
        pmf_value /= Poisson.factorial(k)
        return pmf_value

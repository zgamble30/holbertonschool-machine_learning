#!/usr/bin/env python3

"""
Module to calculate the sum of squared values from 1 to n.
"""


def summation_i_squared(n):
    """
    Calculates the sum of squared values from 1 to n.

    Args:
        n (int): The stopping condition for the summation.

    Returns:
        int: The integer value of the sum.

    If n is not a valid number (less than 1 or not an integer), returns None.
    """
    if isinstance(n, int) and n >= 1:
        return int(n * (n + 1) * (2 * n + 1) / 6)
    else:
        return None

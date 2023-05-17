#!/usr/bin/env python3
"""
Module to calculate the derivative of a polynomial.
"""

def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial.

    Args:
    poly (list): A list of coefficients representing a polynomial.
                 The index of the list represents the power of x that the coefficient belongs to.

    Returns:
    list: A new list of coefficients representing the derivative of the polynomial.
          Returns None if poly is not valid. If the derivative is 0, returns [0].
    """

    # Check if poly is a valid input
    if not isinstance(poly, list) or len(poly) < 2:
        return None

    # Calculate the derivative of the polynomial
    derivative = []
    for i in range(1, len(poly)):
        derivative.append(i * poly[i])

    # Return the calculated derivative or [0] if the derivative is all zeros
    return derivative if derivative != [0] * len(derivative) else [0]


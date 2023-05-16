#!/usr/bin/env python3
"""
Module to calculate the derivative of a polynomial.
"""

def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial.

    Arguments:
    - poly: A list of coefficients representing a polynomial.

    Returns:
    - A new list of coefficients representing the derivative of the polynomial.
    - If poly is not valid, returns None.
    - If the derivative is 0, returns [0].
    """
    if not isinstance(poly, list) or not all(isinstance(coef, (int, float)) for coef in poly):
        return None

    degree = len(poly) - 1
    if degree == 0:
        return [0]

    derivative = []
    for i in range(1, len(poly)):
        derivative.append(poly[i] * i)

    return derivative

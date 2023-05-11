#!/usr/bin/env python3
"""
2-size_me_please module
"""


def matrix_shape(matrix):
    """
    Calculates the shape of matrix and returns list of integers.

    Args:
    matrix (list): A nested list representing the input matrix.

    Returns:
    list: list of integers representing shape of the input matrix.
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape

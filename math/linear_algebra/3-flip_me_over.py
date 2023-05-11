#!/usr/bin/env python3
"""
3-flip_me_over module
"""


def matrix_transpose(matrix):
    """
    Transposes a 2D matrix.

    Args:
    matrix (list): A nested list representing the input matrix.

    Returns:
    list: A nested list representing the transposed matrix.
    """
    transpose = []
    for i in range(len(matrix[0])):
        row = []
        for j in range(len(matrix)):
            row.append(matrix[j][i])
        transpose.append(row)
    return transpose

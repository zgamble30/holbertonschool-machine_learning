#!/usr/bin/env python3
"""
5-across_the_planes module
"""


def add_matrices2D(mat1, mat2):
    """
    Adds two matrices element-wise.

    Args:
    mat1 (list): A nested list representing the first input matrix.
    mat2 (list): A nested list representing the second input matrix.

    Returns:
    list: nested list representing element-wise sum of input matrices.
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    else:
        return [
            [mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))]
            for i in range(len(mat1))
        ]

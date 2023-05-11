#!/usr/bin/env python3
"""
7-gettin_cozy module
"""

def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis without using NumPy.

    Args:
        mat1 (list): A nested list representing the first input matrix.
        mat2 (list): A nested list representing the second input matrix.
        axis (int): An integer representing the axis along which to concatenate the matrices.

    Returns:
        list: A nested list representing the concatenation of the input matrices along the specified axis.
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        else:
            return mat1 + mat2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        else:
            return [mat1[i] + mat2[i] for i in range(len(mat1))]
    else:
        return None

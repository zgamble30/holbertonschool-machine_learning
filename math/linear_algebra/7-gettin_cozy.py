#!/usr/bin/env python3
"""
7-gettin_cozy module
"""

import numpy as np


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis.

    Args:
    mat1 (list): A nested list representing the first input matrix.
    mat2 (list): A nested list representing the second input matrix.
    axis (int): An integer representing the axis along which to concatenate the matrices.

    Returns:
    list: A nested list representing the concatenation of the input matrices along the specified axis.
    """
    try:
        return np.concatenate((mat1, mat2), axis=axis).tolist()
    except ValueError:
        return None

#!/usr/bin/env python3

"""
This module contains a function to perform matrix multiplication without any imports.

The main function in this module is `mat_mul` which multiplies two matrices.
"""

# Your mat_mul function goes here


# Multiplies two matrices mat1 and mat2
# Args: mat1, mat2
# Returns: new matrix
def mat_mul(mat1, mat2):
    """
    Multiplies two matrices mat1 and mat2.
    Args:
        mat1 (list of lists): First matrix
        mat2 (list of lists): Second matrix
    Returns:
        list of lists: Resultant matrix
    """
    # Get dimensions of matrices
    rows1, cols1 = len(mat1), len(mat1[0])
    rows2, cols2 = len(mat2), len(mat2[0])

    # Check if matrices can be multiplied
    if cols1 != rows2:
        return None

    # Create a new matrix to store the result
    result = [[0 for _ in range(cols2)] for _ in range(rows1)]

    # Multiply matrices and store the result in the new matrix
    for i in range(rows1):
        for j in range(cols2):
            for k in range(rows2):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result

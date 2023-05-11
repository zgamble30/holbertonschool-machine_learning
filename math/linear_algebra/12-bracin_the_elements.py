#!/usr/bin/env python3
import numpy as np

def np_elementwise(mat1, mat2):
    """
    This function performs element-wise addition, subtraction, multiplication, and division.

    Parameters:
        mat1 (numpy.ndarray): Input NumPy array.
        mat2 (numpy.ndarray): Input NumPy array.

    Returns:
        tuple: A tuple containing the element-wise sum, difference, product, and quotient, respectively.
    """
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2

    return (add, sub, mul, div)

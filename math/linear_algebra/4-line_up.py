#!/usr/bin/env python3
"""
4-line_up module
"""


def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise.

    Args:
    arr1 (list): A list of ints/floats representing the first input array.
    arr2 (list): A list of ints/floats representing the second input array.

    Returns:
    list: list of ints/floats representing element-wise sum of input arrays.
    """
    if len(arr1) != len(arr2):
        return None
    else:
        return [arr1[i] + arr2[i] for i in range(len(arr1))]

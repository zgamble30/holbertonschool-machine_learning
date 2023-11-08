#!/usr/bin/env python3
"""Performs forward propagation over a
pooling layer of a neural network."""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network.

    Parameters:
    A_prev (np.array): Input array of previous layer
    kernel_shape (tuple): Size of the kernel for pooling (kh, kw)
    stride (tuple, optional): Tuple indicating stride in
    height and width direction. Default is (1, 1)
    mode (str, optional): Pooling mode, can be 'max' or
    'avg'. Default is 'max'

    Returns:
    np.array: Result of the pooling layer
    """

    # Get the dimensions of the input array
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions
    output_height = (h_prev - kh) // sh + 1
    output_width = (w_prev - kw) // sw + 1

    # Initialize output array
    A = np.zeros((m, output_height, output_width, c_prev))

    # Perform the pooling operation
    for h in range(output_height):
        for w in range(output_width):
            for c in range(c_prev):
                pool_start_height = h * sh
                pool_start_width = w * sw
                pool_end_height = pool_start_height + kh
                pool_end_width = pool_start_width + kw

                # Extract the relevant slice of the input array
                a_slice_prev = A_prev[
                    :,
                    pool_start_height:pool_end_height,
                    pool_start_width:pool_end_width,
                    c
                ]

                # Apply pooling mode
                if mode == 'max':
                    A[:, h, w, c] = np.max(a_slice_prev, axis=(1, 2))
                elif mode == 'avg':
                    A[:, h, w, c] = np.mean(a_slice_prev, axis=(1, 2))

    return A

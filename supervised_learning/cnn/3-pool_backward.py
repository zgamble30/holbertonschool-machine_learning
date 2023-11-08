#!/usr/bin/env python3
"""Performs back propagation over a pooling layer of a neural network."""
import numpy as np

def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Perform backpropagation over a pooling layer of a neural network.

    Args:
    - dA: numpy.ndarray of shape (m, h_new, w_new, c_new) containing the partial derivatives
         with respect to the output of the pooling layer
    - A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c) containing the output of the previous layer
    - kernel_shape: tuple of (kh, kw) containing the size of the kernel for the pooling
    - stride: tuple of (sh, sw) containing the strides for the pooling
    - mode: string containing either 'max' or 'avg', indicating whether to perform maximum or average pooling

    Returns:
    - dA_prev: the partial derivatives with respect to the previous layer
    """

    m, h_new, w_new, c_new = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride
    m, h_prev, w_prev, c = A_prev.shape

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for ch in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    if mode == 'max':
                        A_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, ch]
                        mask = (A_slice == np.max(A_slice))
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, ch] += mask * dA[i, h, w, ch]
                    elif mode == 'avg':
                        da = dA[i, h, w, ch]
                        average = da / (kh * kw)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, ch] += np.ones((kh, kw)) * average

    return dA_prev

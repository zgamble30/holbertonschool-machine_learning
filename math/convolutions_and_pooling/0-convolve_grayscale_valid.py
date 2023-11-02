#!/usr/bin/env python3
"""grayscale convolution"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Perform a valid convolution on grayscale images.

    Args:
        images: numpy.ndarray with shape (m, h, w) containing multiple grayscale images.
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel for convolution.

    Returns:
        A numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    output_h = h - kh + 1
    output_w = w - kw + 1
    output = np.zeros((m, output_h, output_w)

    for i in range(output_h):
        for j in range(output_w):
            # Extract the region of interest from images and perform element-wise multiplication
            # Then, sum over axes (1, 2) to get the convolved value
            conv_result = np.sum(images[:, i:i + kh, j:j + kw] * kernel, axis=(1, 2)
            output[:, i, j] = conv_result

    return output

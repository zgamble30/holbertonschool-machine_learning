#!/usr/bin/env python3
"""Convolution on Grayscale Images with
Padding, Stride, and Same/Valid Options"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Perform a convolution on grayscale images with customizable options.

    Args:
    - images (numpy.ndarray): A collection of grayscale images with shape
      (num_images, image_height, image_width).
    - kernel (numpy.ndarray): The convolution kernel with shape
      (kernel_height, kernel_width).
    - padding (str or tuple): Specifies the padding type. Use 'same' for
      same padding, 'valid' for no padding, or provide a tuple (ph, pw)
      for custom padding.
    - stride (tuple): Specifies the stride (sh, sw) for convolution.
      Default is (1, 1).

    Returns:
    - numpy.ndarray: An array containing the convolved images.

    The function takes a set of grayscale images and a convolution kernel
    as input. It performs a convolution with customizable options,
    including padding, stride, and 'same'/'valid' modes.
    The result is a numpy array containing the convolved images.
    """
    num_images, image_height, image_width = images.shape
    kernel_height, kernel_width = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((((image_height - 1) * sh)
              + kernel_height - image_height) // 2) + 1
        pw = ((((image_width - 1) * sw)
              + kernel_width - image_width) // 2) + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    elif type(padding) == tuple:
        ph, pw = padding

    convolved_width = ((image_width - kernel_width + (2 * pw)) // sw) + 1
    convolved_height = ((image_height - kernel_height + (2 * ph)) // sh) + 1

    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    convolved_matrix = np.zeros(
        (num_images, convolved_height, convolved_width)
    )

    for i in range(convolved_width):
        for j in range(convolved_height):
            image_section = padded_images[
                :, sh * j:sh * j + kernel_height, sw * i:sw * i + kernel_width
            ]
            convolved_matrix[:, j, i] = np.tensordot(
                image_section, kernel, axes=2
            )

    return convolved_matrix
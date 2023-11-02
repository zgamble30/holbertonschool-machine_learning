#!/usr/bin/env python3
"""Convolution with Padding on Grayscale Images"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Perform a convolution on grayscale images with custom padding.

    Args:
    - images (numpy.ndarray): A collection of grayscale
    images with shape (num_images, image_height, image_width).
    - kernel (numpy.ndarray): The convolution
    kernel with shape (kernel_height, kernel_width).
    - padding (tuple): A tuple of (padding_height, padding_width).

    Returns:
    - numpy.ndarray: An array containing the
    convolved images.

    The function takes a set of grayscale images,
    a convolution kernel, and custom padding values as input.
    It performs a convolution while applying the specified padding.
    The result is a numpy array containing the convolved images.
    """
    num_images = images.shape[0]
    image_height = images.shape[1]
    image_width = images.shape[2]
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]
    padding_height, padding_width = padding

    # Pad the images with zeros
    padded_images = np.pad(
        images, ((0, 0), (padding_height, padding_height), (padding_width, padding_width)),
        'constant'
    )

    # Calculate the dimensions of the convolved images
    convolved_width = image_width - kernel_width + 1 + (2 * padding_width)
    convolved_height = image_height - kernel_height + 1 + (2 * padding_height)
    convolved_images = np.zeros((num_images, convolved_height, convolved_width))

    for x in range(convolved_width):
        for y in range(convolved_height):
            image_section = padded_images[:, y:y + kernel_height, x:x + kernel_width]
            convolved_images[:, y, x] = np.tensordot(image_section, kernel, axes=2)

    return convolved_images

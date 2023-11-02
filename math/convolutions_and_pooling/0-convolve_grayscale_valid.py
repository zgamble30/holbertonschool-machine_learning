#!/usr/bin/env python3
"""convolution on grayscale images"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Perform a valid convolution on grayscale
    images using a given kernel.

    Args:
    - images (numpy.ndarray): A collection of grayscale images
    with shape (num_images, image_height, image_width).
    - kernel (numpy.ndarray): The convolution kernel with
    shape (kernel_height, kernel_width).

    Returns:
    - numpy.ndarray: An array containing the convolved images.

    The function takes a set of grayscale images
    and a convolution kernel as input.
    It performs a valid convolution, meaning
    the output size is reduced compared to the input.
    The result is a numpy array containing the convolved images.

    Example:
    >>> images = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    >>> kernel = np.array([[1, 0], [0, -1]])
    >>> convolve_grayscale_valid(images, kernel)
    array([[[ 1,  2],
            [ 4,  5],
            [ 7,  8]]])
    """
    num_images = images.shape[0]
    image_height = images.shape[1]
    image_width = images.shape[2]
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    output_width = image_width - kernel_width + 1
    output_height = image_height - kernel_height + 1
    convolved_images = np.zeros((num_images, output_height, output_width)

    for x in range(output_width):
        for y in range(output_height):
            image_section = images[:, y:y + kernel_height, x:x + kernel_width]
            convolved_images[:, y, x] = np.tensordot(
                image_section, kernel, axes=2
            )

    return convolved_images

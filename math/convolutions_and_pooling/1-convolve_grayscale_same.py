#!/usr/bin/env python3
"""Same convolution on grayscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Perform a same convolution on grayscale images using a given kernel.

    Args:
    - images (numpy.ndarray): A collection of grayscale images with
    shape (num_images, image_height, image_width).
    - kernel (numpy.ndarray): The convolution kernel with
    shape (kernel_height, kernel_width).

    Returns:
    - numpy.ndarray: An array containing the convolved images.

    The function takes a set of grayscale
    images and a convolution kernel as input.
    It performs a same convolution, meaning the
    output size is the same as the input.
    The result is a numpy array containing the convolved images.

    Example:
    >>> images = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    >>> kernel = np.array([[1, 0], [0, -1]])
    >>> convolve_grayscale_same(images, kernel)
    array([[[0,  0,  0],
            [0,  1,  2],
            [0,  4,  5]]])
    """
    num_images = images.shape[0]
    image_height = images.shape[1]
    image_width = images.shape[2]
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    # Calculate padding sizes
    pad_height = kernel_height - 1
    pad_width = kernel_width - 1

    # Calculate output size
    output_height = image_height
    output_width = image_width

    # Pad the images with zeros
    pad_values = ((0, 0), (pad_height, pad_height), (pad_width, pad_width))
    padded_images = np.pad(images, pad_values, mode='constant')

    # Initialize the output array
    convolved_images = np.zeros((num_images, output_height, output_width))

    # Perform convolution
    for x in range(output_width):
        for y in range(output_height):
            image_section = padded_images[:, y:y + kernel_height, x:x + kernel_width]
            convolved_images[:, y, x] = np.tensordot(image_section, kernel, axes=2)

    return convolved_images

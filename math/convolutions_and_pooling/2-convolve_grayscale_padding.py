#!/usr/bin/env python3
"""Convolution with Custom Padding on Grayscale Images"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Perform a convolution on grayscale images with custom padding.

    Args:
        images (numpy.ndarray): A collection of
        grayscale images with shape
        (num_images, image_height, image_width).
        kernel (numpy.ndarray): The convolution
        kernel with shape
        (kernel_height, kernel_width).
        padding (tuple): A tuple of (ph, pw)
        representing the padding for the height
        and width of the image.

    Returns:
        numpy.ndarray: An array containing the
        convolved images.

    The function takes a set of grayscale
    images and a convolution kernel as input.
    It performs a convolution with custom padding.
    The result is a numpy array containing the
    convolved images.
    """
    num_images = images.shape[0]
    image_height = images.shape[1]
    image_width = images.shape[2]
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    pad_height = padding[0]
    pad_width = padding[1]

    # Pad the images with zeros
    padded_images = np.pad(
        images, ((0, 0), (pad_height, pad_height), (pad_width, pad_width))
    )

    convolved_images = np.zeros((num_images, image_height, image_width))

    for x in range(image_width):
        for y in range(image_height):
            image_section = padded_images[:, y:y + kernel_height, x:x + kernel_width]
            if image_section.shape[1:] == kernel.shape:
                convolved_images[:, y, x] = np.tensordot(
                    image_section, kernel, axes=2
                )

    return convolved_images

#!/usr/bin/env python3
"""performs pooling"""
import numpy as np

def pool(images, kernel_shape, stride, mode='max'):
    """
    Apply pooling to multi-channel images with customizable options.

    Args:
    - images (numpy.ndarray): A collection of multi-channel images with shape
      (num_samples, image_height, image_width, num_channels).
    - kernel_shape (tuple): The shape of the pooling kernel (kh, kw).
    - stride (tuple): Specifies the stride (sh, sw) for pooling.
    - mode (str): Specifies the pooling mode, either 'max' or 'average'.
      Default is 'max'.

    Returns:
    - numpy.ndarray: An array containing the pooled multi-channel images.

    This function applies pooling to multi-channel images using a specified
    pooling kernel and customizable options, including stride and pooling mode.
    The result is a numpy array containing the pooled multi-channel images.
    """
    num_samples, input_height, input_width, num_channels = images.shape
    kernel_height, kernel_width = kernel_shape
    sh, sw = stride

    if mode == "max":
        pool_operation = np.amax
    else:
        pool_operation = np.average

    pad_height, pad_width = 0, 0
    pooled_width = (input_width - kernel_width + (2 * pad_width)) // sw + 1
    pooled_height = (input_height - kernel_height + (2 * pad_height)) // sh + 1
    pooled_images = np.zeros((num_samples, pooled_height, pooled_width, num_channels))

    for i in range(pooled_height):
        for j in range(pooled_width):
            pooled_images[:, i, j, :] = pool_operation(
                images[:, sh * i:sh * i + kernel_height, sw * j:sw * j + kernel_width, :],
                axis=(1, 2)
            )

    return pooled_images

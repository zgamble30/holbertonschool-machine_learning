#!/usr/bin/env python3

import numpy as np

def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Perform convolution on multi-channel images with customizable options.

    Args:
    - images (numpy.ndarray): A collection of multi-channel images with shape
      (num_samples, image_height, image_width, num_channels).
    - kernels (numpy.ndarray): The convolution kernels with shape
      (kernel_height, kernel_width, num_input_channels, num_output_channels).
    - padding (str or tuple): Specifies the padding type. Use 'same' for
      same padding, 'valid' for no padding, or provide a tuple (ph, pw)
      for custom padding.
    - stride (tuple): Specifies the stride (sh, sw) for convolution.
      Default is (1, 1).

    Returns:
    - numpy.ndarray: An array containing the convolved multi-channel images.

    This function performs convolution on multi-channel images using the
    specified convolution kernels and customizable options, including
    padding, stride, and 'same'/'valid' modes. The result is a numpy array
    containing the convolved multi-channel images.
    """
    num_samples, input_height, input_width, num_input_channels = images.shape
    kernel_height, kernel_width, num_input_channels, num_output_channels = kernels.shape
    sh, sw = stride

    if padding == 'same':
        pad_height = ((((input_height - 1) * sh) + kernel_height - input_height) // 2) + 1
        pad_width = ((((input_width - 1) * sw) + kernel_width - input_width) // 2) + 1
    elif padding == 'valid':
        pad_height = 0
        pad_width = 0
    elif isinstance(padding, tuple) and len(padding) == 2:
        pad_height, pad_width = padding
    else:
        raise ValueError("Invalid padding parameter. Use 'same', 'valid', or a (ph, pw) tuple.")

    output_width = ((input_width + (2 * pad_width) - kernel_width) // sw) + 1
    output_height = ((input_height + (2 * pad_height) - kernel_height) // sh) + 1

    padded_images = np.pad(images, ((0, 0), (pad_height, pad_height), (pad_width, pad_width), (0, 0)), 'constant')
    convolved_images = np.zeros((num_samples, output_height, output_width, num_output_channels))

    for output_channel in range(num_output_channels):
        for x in range(output_width):
            for y in range(output_height):
                image_section = padded_images[:, sh * y:sh * y + kernel_height, sw * x:sw * x + kernel_width, :]
                convolved_images[:, y, x, output_channel] = np.tensordot(image_section, kernels[:, :, :, output_channel], axes=3)

    return convolved_images

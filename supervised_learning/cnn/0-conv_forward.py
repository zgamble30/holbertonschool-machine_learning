#!/usr/bin/env python3
""" Performs a custom convolution operation."""
import numpy as np


def custom_convolution(A, K, b, act, pad_type="same", step=(1, 1)):
    """
    Performs a custom convolution operation.
    
    :param A: Input array
    :param K: Kernel array
    :param b: Bias array
    :param act: Activation function
    :param pad_type: Padding type, either "same" or "valid"
    :param step: Step size for convolution (also known as stride)
    :return: Convolution result with activation applied
    """

    # Get the dimensions of the input and kernel arrays
    m, h_i, w_i, c_i = A.shape
    k_h, k_w, c_i, c_o = K.shape
    s_h, s_w = step

    # Calculate padding considering the padding type
    if pad_type == "same":
        p_h = int(np.ceil(((h_i - 1) * s_h + k_h - h_i) / 2))
        p_w = int(np.ceil(((w_i - 1) * s_w + k_w - w_i) / 2))
    else:
        p_h = p_w = 0

    # Apply padding to the input array
    A_pad = np.pad(A, ((0, 0), (p_h, p_h), (p_w, p_w), (0, 0)), 'constant')

    # Compute the dimensions of the convolution output
    conv_h = (h_i + 2 * p_h - k_h) // s_h + 1
    conv_w = (w_i + 2 * p_w - k_w) // s_w + 1

    # Initialize the convolution output array
    conv_output = np.zeros((m, conv_h, conv_w, c_o))

    # Perform the convolution operation
    for i in range(conv_h):
        for j in range(conv_w):
            for c in range(c_o):
                f_h_start = i * s_h
                f_w_start = j * s_w
                f_h_end = f_h_start + k_h
                f_w_end = f_w_start + k_w

                # Extract the relevant slice of the input array and the kernel
                A_slice = A_pad[:, f_h_start:f_h_end, f_w_start:f_h_end, :]
                K_slice = K[:, :, :, c]
                biases = b[0, 0, 0, c]

                # Apply the convolution operation and add the bias
                conv_output[:, i, j, c] = np.sum(A_slice * K_slice, axis=(1, 2, 3)) + biases

    # Apply the activation function to the convolution output
    A_out = act(conv_output)
    
    return A_out

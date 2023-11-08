#!/usr/bin/env python3
""" Performs a custom convolution operation."""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward convolution operation.
    
    Parameters:
    A_prev (np.array): Input array of previous layer
    W (np.array): Weights array
    b (np.array): Bias array
    activation (function): Activation function to be used
    padding (str, optional): Type of padding, can be "same" or "valid". Default is "same"
    stride (tuple, optional): Tuple indicating stride in height and width direction. Default is (1, 1)
    
    Returns:
    np.array: Result of the convolution operation
    """
    
    # Get the dimensions of the input and weights arrays
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    # Calculate padding based on padding type
    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph = pw = 0

    # Calculate output dimensions
    output_height = (h_prev + 2 * ph - kh) // sh + 1
    output_width = (w_prev + 2 * pw - kw) // sw + 1

    # Initialize output array
    Z = np.zeros((m, output_height, output_width, c_new))
    
    # Pad input array
    A_prev_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')

    # Perform the convolution operation
    for h in range(output_height):
        for w in range(output_width):
            for c in range(c_new):
                filter_start_height = h * sh
                filter_start_width = w * sw
                filter_end_height = filter_start_height + kh
                filter_end_width = filter_start_width + kw

                # Extract the relevant slice of the input array and the weights
                a_slice_prev = A_prev_pad[:, filter_start_height:filter_end_height, filter_start_width:filter_end_width, :]
                weights = W[:, :, :, c]

                # Adjust the dimensions of the bias term for broadcasting
                biases = b[:, :, :, c]

                # Apply the convolution operation and add the bias
                Z[:, h, w, c] = np.sum(a_slice_prev * weights, axis=(1, 2, 3)) + biases

    # Apply the activation function to the convolution output
    A = activation(Z)
    
    return A

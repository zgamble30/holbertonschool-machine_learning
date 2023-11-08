#!/usr/bin/env python3
"""Performs back propagation over a convolutional layer of a neural network."""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network.

    Parameters:
    dZ (np.array): Partial derivatives with respect to the unactivated output
    A_prev (np.array): Output of the previous layer
    W (np.array): Kernels for the convolution
    b (np.array): Biases applied to the convolution
    padding (str, optional): Type of padding, can be "same" or "valid". Default is "same"
    stride (tuple, optional): Tuple indicating stride in height and width direction. Default is (1, 1)

    Returns:
    tuple: Partial derivatives with respect to the previous layer (dA_prev),
           the kernels (dW), and the biases (db)
    """

    # Get the dimensions of the input, kernels, and output arrays
    m, h_new, w_new, c_new = dZ.shape
    _, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    # Initialize output arrays
    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    # Pad input array
    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph = pw = 0

    A_prev_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        'constant'
    )

    # Loop over examples, output height, output width, and channels
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    # Calculate the starting and ending positions for the current slice
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    # Extract the slice from A_prev_pad
                    a_slice = A_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :]

                    # Update gradients using the chain rule and the gradient of the activation function
                    dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

    return dA_prev, dW, db

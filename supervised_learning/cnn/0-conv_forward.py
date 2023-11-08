#!/usr/bin/env python3
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Perform forward propagation over a convolutional layer of a neural network.

    Args:
    - A_prev (numpy.ndarray): Output of the previous layer with shape (m, h_prev, w_prev, c_prev).
    - W (numpy.ndarray): Kernels for the convolution with shape (kh, kw, c_prev, c_new).
    - b (numpy.ndarray): Biases applied to the convolution with shape (1, 1, 1, c_new).
    - activation (function): Activation function applied to the convolution.
    - padding (str): Type of padding used, either "same" or "valid". Default is "same".
    - stride (tuple): Strides for the convolution (sh, sw).

    Returns:
    - numpy.ndarray: Output of the convolutional layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = max(((h_prev - 1) * sh + kh - h_prev) // 2, 0)
        pw = max(((w_prev - 1) * sw + kw - w_prev) // 2, 0)
    elif padding == "valid":
        ph, pw = 0, 0

    padded_A_prev = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')

    conv_h = (h_prev - kh + (2 * ph)) // sh + 1
    conv_w = (w_prev - kw + (2 * pw)) // sw + 1

    conv_output = np.zeros((m, conv_h, conv_w, c_new))

    for i in range(conv_h):
        for j in range(conv_w):
            image_section = padded_A_prev[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            conv_output[:, i, j, :] = activation(np.sum(image_section * W, axis=(1, 2, 3)) + b)

    return conv_output

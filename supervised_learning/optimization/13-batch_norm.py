#!/usr/bin/env python3
"""
Defines a function batch_norm
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of
    a neural network using batch normalization
    """
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    Z_norm = gamma * Z_norm + beta
    return Z_norm

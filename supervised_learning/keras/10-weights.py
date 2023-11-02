#!/usr/bin/env python3
"""
configs network
"""
import tensorflow as tf


def save_weights(network, filename, save_format='h5'):
    """
    Save the weights of a Keras model to a file.
    
    Args:
    - network: Keras model whose
    weights should be saved.
    - filename: The path of the file
    where the weights should be saved.
    - save_format: The format in which
    the weights should be saved. Default is 'h5'.
    
    Returns:
    - None
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """
    Load weights from a file and set them to a Keras model.
    
    Args:
    - network: Keras model to which
    the weights should be loaded.
    - filename: The path of the file
    from which the weights should be loaded.
    
    Returns:
    - None
    """
    network.load_weights(filename)

"""Example usage:
Assuming you have a Keras model named 'network'
and you want to save its weights to 'weights2.h5'
and later load these weights back to another Keras model."""

# Saving weights to a file
save_weights(network, 'weights2.h5')

# Loading weights from a file
load_weights(network2, 'weights2.h5')

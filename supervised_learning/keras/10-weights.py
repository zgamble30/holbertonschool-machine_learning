#!/usr/bin/env python3
"""
saves and loads weights
"""
import tensorflow.keras as K


# Function to save model weights to a file
def save_weights(network, filename, save_format='h5'):
    """
    Save the weights of a Keras model to a file.

    Args:
        network (tf.keras.Model): The Keras
        model whose weights you want to save.
        filename (str): The name of the file to
        save the weights to.
        save_format (str): The format in which
        to save the weights (default is 'h5').

    Returns:
        None
    """
    network.save_weights(filename, save_format=save_format)


# Function to load model weights from a file
def load_weights(network, filename):
    """
    Load weights from a file and set them to a Keras model.

    Args:
        network (tf.keras.Model): The Keras
        model to which you want to load the weights.
        filename (str): The name of the
        file containing the weights to be loaded.

    Returns:
        None
    """
    network.load_weights(filename)

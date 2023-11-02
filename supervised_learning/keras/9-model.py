#!/usr/bin/env python3
"""
saves and loads models
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Save an entire Keras model to a file.

    Args:
        network (tf.keras.Model): The model to save.
        filename (str): The path to the file
        where the model should be saved.

    Returns:
        None
    """
    network.save(filename)


def load_model(filename):
    """
    Load an entire Keras model from a file.

    Args:
        filename (str): The path to the file
        from which to load the model.

    Returns:
        tf.keras.Model: The loaded Keras model.
    """
    loaded_model = K.models.load_model(filename)
    return loaded_model

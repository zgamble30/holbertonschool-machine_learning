#!/usr/bin/env python3
"""
configs network
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model's configuration in a text file.

    Args:
        - network: the model whose configuration
        should be saved.
        - filename: the path of the file that
        the configuration should be saved to.

    Returns: None
    """
    config = network.to_json()
    with open(filename, 'w') as file:
        file.write(config)


def load_config(filename):
    """
    Loads a model with a specific configuration.

    Args:
        - filename: the path of the file
        containing the model's configuration.

    Returns: the loaded model
    """
    with open(filename, 'r') as file:
        config = file.read()
    loaded_model = K.models.model_from_json(config)
    return loaded_model

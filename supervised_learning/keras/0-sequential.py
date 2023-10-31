#!/usr/bin/env python3
"""
Builds a Keras model with a specified architecture
"""
import tensorflow.keras as K

def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a Keras model with the specified architecture.

    Args:
    - nx: number of input features to the network.
    - layers: list containing the number of nodes in each layer of the network.
    - activations: list containing the activation functions used for each layer of the network.
    - lambtha: L2 regularization parameter.
    - keep_prob: probability that a node will be kept for dropout.

    Returns:
    A Keras model.
    """
    model = K.Sequential()
    L2 = K.regularizers.l2(lambtha)

    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(layers[i], input_dim=nx,
                                     activation=activations[i],
                                     kernel_regularizer=L2))
        else:
            model.add(K.layers.Dense(layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=L2))
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model

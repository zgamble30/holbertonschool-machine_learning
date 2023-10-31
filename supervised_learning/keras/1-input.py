#!/usr/bin/env python3
"""
Builds a Keras model with a specified architecture using Input class.
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a Keras model with the specified architecture using Input class.

    Args:
    - nx: number of input features to the network.
    - layers: list containing the number of nodes in each layer of the network.
    - activations: list containing the
    activation functions used for each layer of the network.
    - lambtha: L2 regularization parameter.
    - keep_prob: probability that a node will be kept for dropout.

    Returns:
    A Keras model.
    """
    X = K.Input(shape=(nx,))
    regularizer = K.regularizers.l2(lambtha)
    A = X

    for i in range(len(layers)):
        layer = K.layers.Dense(units=layers[i], activation=activations[i],
                             kernel_regularizer=regularizer)(A)

        if i < len(layers) - 1 and keep_prob < 1.0:
            dropout = K.layers.Dropout(1 - keep_prob)(layer)
            A = dropout
        else:
            A = layer

    model = K.models.Model(inputs=X, outputs=A)
    return model

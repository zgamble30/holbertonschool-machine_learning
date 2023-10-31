#!/usr/bin/env python3
"""
Train Model
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient
    descent with optional validation data.

    Args:
    - network: the model to train.
    - data: a numpy.ndarray of shape (m, nx)
      containing the input data.
    - labels: a one-hot numpy.ndarray of shape (m, classes)
      containing the labels of data.
    - batch_size: the size of the batch
    used for mini-batch gradient descent.
    - epochs: the number of passes
    through data for mini-batch gradient descent.
    - validation_data: a tuple (X_valid, Y_valid)
      containing validation data, where X_valid is the input data
      and Y_valid is the labels for validation. Default is None.
    - verbose: a boolean that determines if
      output should be printed during training.
    - shuffle: a boolean that determines whether
      to shuffle the batches every epoch.

    Returns:
    - the History object generated after training the model.
    """
    if validation_data is not None:
        X_valid, Y_valid = validation_data
        validation_data = (X_valid, Y_valid)

    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data
    )

    return history

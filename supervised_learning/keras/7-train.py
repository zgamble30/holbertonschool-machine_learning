#!/usr/bin/env python3
"""
save best iteration
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1, verbose=True,
                shuffle=False):
    """
    Trains a model using mini-batch gradient descent with optional validation data,
    early stopping, and learning rate decay.

    Args:
        - network: the model to train.
        - data: a numpy.ndarray of shape (m, nx) containing the input data.
        - labels: a one-hot numpy.ndarray of shape
          (m, classes) containing the labels of data.
        - batch_size: the size of the batch used for mini-batch gradient descent.
        - epochs: the number of passes through data for mini-batch gradient descent.
        - validation_data: a tuple (X_valid, Y_valid) containing validation data,
          where X_valid is the input data and Y_valid is the labels for validation.
          Default is None.
        - early_stopping: a boolean that indicates whether early stopping should be used.
          Default is False.
        - patience: the patience used for early stopping. Default is 0.
        - learning_rate_decay: a boolean that indicates whether learning rate decay should be used.
          Default is False.
        - alpha: the initial learning rate. Default is 0.1.
        - decay_rate: the decay rate for the learning rate. Default is 1.
        - verbose: a boolean that determines if output should be printed during training.
        - shuffle: a boolean that determines whether to shuffle the batches every epoch.

    Returns:
        - the History object generated after training the model.
    """
    if validation_data is not None:
        X_valid, Y_valid = validation_data
        validation_data = (X_valid, Y_valid)

    callbacks = []

    if early_stopping:
        early_stopping_callback = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )
        callbacks.append(early_stopping_callback)

    if learning_rate_decay:
        def lr_schedule(epoch):
            return alpha / (1 + decay_rate * epoch)
        
        learning_rate_scheduler = K.callbacks.LearningRateScheduler(lr_schedule)
        callbacks.append(learning_rate_scheduler)

    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks
    )

    return history

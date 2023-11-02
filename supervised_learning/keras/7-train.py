#!/usr/bin/env python3
"""
also trains with learning rate decay
"""
import tensorflow.keras as K


class LearningRateScheduler(K.callbacks.Callback):
    def __init__(self, alpha, decay_rate):
        self.alpha = alpha
        self.decay_rate = decay_rate
        self.epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch += 1
        self.alpha = self.alpha / (1 + self.epoch * self.decay_rate)
        K.backend.set_value(self.model.optimizer.lr, self.alpha)
        print(f"\nEpoch {self.epoch}/{self.params['epochs']}"
              f"\n\nEpoch {self.epoch:05d}: LearningRateScheduler setting learning rate to {self.alpha}.")

def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1, verbose=True, shuffle=False):
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
        lr_scheduler = LearningRateScheduler(alpha, decay_rate)
        callbacks.append(lr_scheduler)

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

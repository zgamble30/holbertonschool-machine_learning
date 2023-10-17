#!/usr/bin/env python3
"""
Defines a function create_Adam_op
"""

import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Creates the training operation for a neural network in tensorflow
    using the Adam optimization algorithm
    """
    optimizer = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    train_op = optimizer.minimize(loss)

    return train_op

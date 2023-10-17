#!/usr/bin/env python3
"""RMSProp Upgraded"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Creates the training operation for a neural network in TensorFlow
    using the RMSProp optimization algorithm."""
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                          decay=beta2,
                                          epsilon=epsilon)
    return optimizer.minimize(loss)

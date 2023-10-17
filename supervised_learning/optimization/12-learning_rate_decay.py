#!/usr/bin/env python3
"""
Defines a function learning_rate_decay
"""

import tensorflow.compat.v1 as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Creates a learning rate decay operation in tensorflow using inverse time decay
    """
    learning_rate = tf.train.inverse_time_decay(
        alpha, global_step, decay_step, decay_rate, staircase=True
    )
    return learning_rate

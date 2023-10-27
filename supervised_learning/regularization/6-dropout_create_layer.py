#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout

def dropout_create_layer(prev, n, activation, keep_prob):
    reg = Dropout(rate=keep_prob)
    init = tf.keras.initializers.VarianceScaling(mode="fan_avg")

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init
    )(prev)
    output = reg(layer)

    return output


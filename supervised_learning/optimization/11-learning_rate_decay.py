#!/usr/bin/env python3
"""
Defines a function learning_rate_decay
"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in numpy
    """
    updated_alpha = alpha / (1 + decay_rate * (global_step // decay_step))
    return updated_alpha

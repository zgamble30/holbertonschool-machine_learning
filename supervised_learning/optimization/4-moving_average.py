#!/usr/bin/env python3
"""Moving Average"""

def moving_average(data, beta):
    """Calculates the weighted moving average of a data set."""
    m_avg = []
    vt = 0

    for t in range(1, len(data) + 1):
        vt = beta * vt + (1 - beta) * data[t - 1]
        corrected_bias = vt / (1 - beta**t)
        m_avg.append(corrected_bias)

    return m_avg

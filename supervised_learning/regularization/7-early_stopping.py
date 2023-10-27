#!/usr/bin/env python3
"""
early stopping determined by gradient descent
"""
def early_stopping(current_cost, best_cost, threshold, max_patience, patience_count):
    """
    A function utilizing Early Stopping Regularization, The Easiest
    One To Implement
    """
    if (best_cost - current_cost) > threshold:
        patience_count = 0
    else:
        patience_count = patience_count + 1

    if patience_count >= max_patience:
        return (True, patience_count)
    else:
        return (False, patience_count)

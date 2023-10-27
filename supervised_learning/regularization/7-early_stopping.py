#!/usr/bin/env python3
"""
early stopping determined by gradient descent
"""
def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if you should stop gradient descent early.

    Args:
        cost (float): The current validation cost of the neural network.
        opt_cost (float): The lowest recorded validation cost of the neural network.
        threshold (float): The threshold used for early stopping.
        patience (int): The patience count used for early stopping.
        count (int): The count of how long the threshold has not been met.

    Returns:
        A tuple (boolean, updated_count) where the boolean indicates whether
        the network should be stopped early, and updated_count is the updated count.
    """

    # Check if the current cost is greater than the optimal cost plus the threshold
    if cost > opt_cost + threshold:
        count += 1
    else:
        # Reset the count if the cost is below the threshold
        count = 0

    # Check if the count has reached the patience limit
    if count >= patience:
        return True, count
    else:
        return False, count

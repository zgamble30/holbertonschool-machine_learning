#!/usr/bin/env python3
"""Evaluation Station"""
import tensorflow.compat.v1 as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop


def evaluate(X, Y, save_path):
    """
    Evaluates the output of a neural network.

    Arguments:
    - X: numpy.ndarray containing the input data to evaluate
    - Y: numpy.ndarray containing the one-hot labels for X
    - save_path: location to load the model from

    Returns: the network’s prediction, accuracy, and loss, respectively
    """

    # Create placeholders
    x, y = create_placeholders(X.shape[1], Y.shape[1])

    # Forward propagation
    y_pred = forward_prop(x, [], [])

    # Calculate accuracy
    accuracy = calculate_accuracy(y, y_pred)

    # Calculate loss
    loss = calculate_loss(y, y_pred)

    # Add to the graph's collection
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)

    # Initialize global variables
    init = tf.global_variables_initializer()

    # Start session to execute computation graph
    with tf.Session() as sess:
        sess.run(init)

        # Restore model
        saver = tf.train.Saver()
        saver.restore(sess, save_path)

        # Evaluate on test data
        pred, acc, cst = sess.run([y_pred, accuracy, loss], feed_dict={x: X, y: Y})

    return pred, acc, cst

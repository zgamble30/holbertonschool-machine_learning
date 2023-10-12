#!/usr/bin/env python3
"""Evaluation Station"""
import tensorflow.compat.v1 as tf

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
    x = tf.placeholder(tf.float32, shape=[None, X.shape[1]], name='x')
    y = tf.placeholder(tf.float32, shape=[None, Y.shape[1]], name='y')

    # Forward propagation
    layer = x
    for i in range(len(layer_sizes) - 1):
        layer = tf.layers.Dense(layer_sizes[i], activation=activations[i], \
                                kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg'), \
                                name='layer')(layer)
    y_pred = tf.layers.Dense(layer_sizes[-1], activation=None, name='layer')(layer)

    # Calculate accuracy
    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='Mean')

    # Calculate loss
    loss = tf.losses.softmax_cross_entropy(y, y_pred, scope='softmax_cross_entropy_loss')

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

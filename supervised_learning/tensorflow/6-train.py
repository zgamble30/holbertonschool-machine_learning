#!/usr/bin/env python3

import tensorflow.compat.v1 as tf


# Importing functions from the other modules using assignment
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op

def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier.

    Args:
        X_train (numpy.ndarray): Training input data.
        Y_train (numpy.ndarray): Training labels.
        X_valid (numpy.ndarray): Validation input data.
        Y_valid (numpy.ndarray): Validation labels.
        layer_sizes (list): Number of nodes in each layer of the network.
        activations (list): Activation functions for each layer.
        alpha (float): Learning rate.
        iterations (int): Number of iterations to train over.
        save_path (str): Path to save the model.

    Returns:
        str: Path where the model was saved.
    """
    tf.reset_default_graph()

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(iterations + 1):
            train_cost, train_accuracy = sess.run([loss, accuracy],
                                                  feed_dict={x: X_train, y: Y_train})

            valid_cost, valid_accuracy = sess.run([loss, accuracy],
                                                  feed_dict={x: X_valid, y: Y_valid})

            if i % 100 == 0 or i == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(train_cost))
                print("\tTraining Accuracy: {}".format(train_accuracy))
                print("\tValidation Cost: {}".format(valid_cost))
                print("\tValidation Accuracy: {}".format(valid_accuracy))

            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})

        saver = tf.train.Saver()
        saved_path = saver.save(sess, save_path)

    return saved_path

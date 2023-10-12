#!/usr/bin/env python3
import tensorflow as tf
import numpy as np

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
    # Reset the TensorFlow graph
    tf.reset_default_graph()

    # Create placeholders
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])

    # Forward propagation
    y_pred = forward_prop(x, layer_sizes, activations)

    # Calculate accuracy and loss
    accuracy = calculate_accuracy(y, y_pred)
    loss = calculate_loss(y, y_pred)

    # Create training operation
    train_op = create_train_op(loss, alpha)

    # Add tensors to graph's collection
    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)
    tf.add_to_collection("y_pred", y_pred)
    tf.add_to_collection("loss", loss)
    tf.add_to_collection("accuracy", accuracy)
    tf.add_to_collection("train_op", train_op)

    # Initialize global variables
    init = tf.global_variables_initializer()

    # Start session to execute computation graph
    with tf.Session() as sess:
        sess.run(init)

        for i in range(iterations + 1):
            t_cost, t_accuracy = sess.run([loss, accuracy],
                                          feed_dict={x: X_train, y: Y_train})
            v_cost, v_accuracy = sess.run([loss, accuracy],
                                          feed_dict={x: X_valid, y: Y_valid})

            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(t_cost))
                print("\tTraining Accuracy: {}".format(t_accuracy))
                print("\tValidation Cost: {}".format(v_cost))
                print("\tValidation Accuracy: {}".format(v_accuracy))

            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})

        # Save model
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_path)

    return save_path

#!/usr/bin/env python3
"""Train module"""
import tensorflow.compat.v1 as tf
from typing import Tuple


calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train: np.ndarray, Y_train: np.ndarray,
          X_valid: np.ndarray, Y_valid: np.ndarray,
          layer_sizes: List[int], activations: List[Callable],
          alpha: float, iterations: int,
          save_path: str = "/tmp/model.ckpt") -> str:
    """
    Builds, trains, and saves a neural network classifier.

    Args:
        X_train: Numpy.ndarray - Training input data.
        Y_train: Numpy.ndarray - Training labels.
        X_valid: Numpy.ndarray - Validation input data.
        Y_valid: Numpy.ndarray - Validation labels.
        layer_sizes: List[int] - Number of nodes in each layer.
        activations: List[Callable] - Activation functions for each layer.
        alpha: float - Learning rate.
        iterations: int - Number of iterations to train over.
        save_path: str - Designates where to save the model.

    Returns:
        str: The path where the model was saved.
    """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    tf.add_to_collection('placeholders', x)
    tf.add_to_collection('placeholders', y)
    tf.add_to_collection('tensors', y_pred)
    tf.add_to_collection('tensors', loss)
    tf.add_to_collection('tensors', accuracy)
    tf.add_to_collection('operation', train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(iterations + 1):
            cost_train = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            acc_train = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            cost_valid = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            acc_valid = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})

            if i % 100 == 0 or i == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(cost_train))
                print("\tTraining Accuracy: {}".format(acc_train))
                print("\tValidation Cost: {}".format(cost_valid))
                print("\tValidation Accuracy: {}".format(acc_valid))

            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})

        save_path = saver.save(sess, save_path)
        return save_path

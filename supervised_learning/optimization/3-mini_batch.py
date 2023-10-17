#!/usr/bin/env python3
"""Mini-Batch"""

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
shuffle_data = __import__('2-shuffle_data').shuffle_data

def create_placeholders(nx, classes):
    """Creates two placeholders."""
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    return x, y

def create_layer(prev, n, activation):
    """Creates a layer."""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation, kernel_initializer=init, name='layer')(prev)
    return layer

def forward_prop(x, layer_sizes, activations):
    """Creates the forward propagation graph for the neural network."""
    prev = x
    for i in range(len(layer_sizes)):
        prev = create_layer(prev, layer_sizes[i], activations[i])
    return prev

def calculate_accuracy(y, y_pred):
    """Calculates the accuracy of a prediction."""
    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='Mean')
    return accuracy

def calculate_loss(y, y_pred):
    """Calculates the softmax cross-entropy loss of a prediction."""
    loss = tf.losses.softmax_cross_entropy(y, y_pred, scope='softmax_cross_entropy_loss')
    return loss

def create_train_op(loss, alpha):
    """Creates the training operation for the network."""
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train_op = optimizer.minimize(loss)
    return train_op

def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5, load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    """Trains a loaded neural network model using mini-batch gradient descent."""
    nx = X_train.shape[1]
    classes = Y_train.shape[1]
    ops.reset_default_graph()

    x, y = create_placeholders(nx, classes)
    y_pred = forward_prop(x, [256, 256, 10], [tf.nn.tanh, tf.nn.tanh, None])
    accuracy = calculate_accuracy(y, y_pred)
    loss = calculate_loss(y, y_pred)
    train_op = create_train_op(loss, 0.01)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, load_path)

        for epoch in range(epochs + 1):
            X_train, Y_train = shuffle_data(X_train, Y_train)
            t_cost, t_accuracy = sess.run([loss, accuracy], feed_dict={x: X_train, y: Y_train})
            v_cost, v_accuracy = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

            if epoch % 100 == 0 or epoch == epochs:
                print("After {} epochs:".format(epoch))
                print("\tTraining Cost: {}".format(t_cost))
                print("\tTraining Accuracy: {}".format(t_accuracy))
                print("\tValidation Cost: {}".format(v_cost))
                print("\tValidation Accuracy: {}".format(v_accuracy))

            for step in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[step: step + batch_size]
                Y_batch = Y_train[step: step + batch_size]
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})
                if step % 100 == 0:
                    step_cost, step_accuracy = sess.run([loss, accuracy], feed_dict={x: X_batch, y: Y_batch})
                    print("\tStep {}:".format(step))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_accuracy))

        save_path = saver.save(sess, save_path)

    return save_path

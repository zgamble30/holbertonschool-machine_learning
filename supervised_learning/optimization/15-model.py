#!/usr/bin/env python3
"""
Defines a function model
"""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy as np

def forward_prop(x, layer_sizes, activations, epsilon):
    """
    Creates the forward propagation graph for the neural network model
    """
    A = x
    for i in range(len(layer_sizes)):
        A_prev = A
        if i == len(layer_sizes) - 1:
            activation = None
        else:
            activation = activations[i]
        A = create_batch_norm_layer(A_prev, layer_sizes[i], activation, epsilon)
    return A

def create_placeholders(nx, classes):
    """
    Creates two placeholders, x and y, for the neural network
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y

def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way
    """
    m = X.shape[0]
    permutation = list(np.random.permutation(m))
    X_shuffled = X[permutation]
    Y_shuffled = Y[permutation]
    return X_shuffled, Y_shuffled

def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    Builds, trains, and saves a neural network model in tensorflow using
    Adam optimization, mini-batch gradient descent, learning rate decay, and
    batch normalization
    """
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layers, activations, epsilon)

    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    accuracy = tf.metrics.accuracy(tf.argmax(y, 1), tf.argmax(y_pred, 1))

    global_step = tf.Variable(0, trainable=False)
    decay_steps = X_train.shape[0] // batch_size
    alpha = learning_rate_decay(alpha, decay_rate, global_step, decay_steps)
    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs + 1):
            print("After {} epochs:".format(epoch))
            train_cost, train_accuracy = sess.run([loss, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_accuracy = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if epoch < epochs:
                X_train, Y_train = shuffle_data(X_train, Y_train)

            for step in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[step: step + batch_size]
                Y_batch = Y_train[step: step + batch_size]

                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})
                step_cost, step_accuracy = sess.run([loss, accuracy], feed_dict={x: X_batch, y: Y_batch})

                if step % 100 == 0:
                    print("\tStep {}:".format(step))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_accuracy))

        saver.save(sess, save_path)
    return save_path

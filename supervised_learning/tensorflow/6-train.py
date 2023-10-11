#!/usr/bin/env python3
"""
Module containing a function that builds, trains, and saves a neural network classifier.
"""

import tensorflow.compat.v1 as tf

def one_hot(Y, classes):
    """Convert an array to a one-hot matrix."""
    one_hot_matrix = np.zeros((Y.shape[0], classes))
    one_hot_matrix[np.arange(Y.shape[0]), Y] = 1
    return one_hot_matrix

def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier.

    Args:
        X_train (numpy.ndarray): Training input data.
        Y_train (numpy.ndarray): Training labels.
        X_valid (numpy.ndarray): Validation input data.
        Y_valid (numpy.ndarray): Validation labels.
        layer_sizes (list): Number of nodes in each layer of the network.
        activations (list): Activation functions for each layer of the network.
        alpha (float): Learning rate.
        iterations (int): Number of iterations to train over.
        save_path (str): Path to save the model.

    Returns:
        str: The path where the model was saved.
    """
    tf.set_random_seed(0)
    
    # Create placeholders
    x = tf.placeholder(tf.float32, shape=(None, X_train.shape[1]), name="x")
    y = tf.placeholder(tf.float32, shape=(None, Y_train.shape[1]), name="y")

    # Forward propagation
    y_pred = x
    for i in range(len(layer_sizes)):
        y_pred = tf.layers.dense(
            inputs=y_pred,
            units=layer_sizes[i],
            activation=activations[i],
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"),
            name='layer'
        )

    # Placeholder for the loss
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)

    # Placeholder for the accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.math.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1)), tf.float32))

    # Placeholder for the training operation
    train_op = tf.train.GradientDescentOptimizer(alpha).minimize(loss)

    # Add placeholders, tensors, and operation to the graph's collection
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    # Start training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(iterations + 1):
            # Training
            _, train_cost, train_accuracy = sess.run([train_op, loss, accuracy], feed_dict={x: X_train, y: Y_train})

            # Validation (every 100 iterations)
            if i % 100 == 0 or i == 0 or i == iterations:
                valid_cost, valid_accuracy = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {train_cost}")
                print(f"\tTraining Accuracy: {train_accuracy}")
                print(f"\tValidation Cost: {valid_cost}")
                print(f"\tValidation Accuracy: {valid_accuracy}")

        # Save the model
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_path)

    print(f"Model saved in path: {save_path}")
    return save_path


"""if __name__ == '__main__':
    lib = np.load('../data/MNIST.npz')
    X_train_3D = lib['X_train']
    Y_train = lib['Y_train']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1))
    Y_train_oh = one_hot(Y_train, 10)
    X_valid_3D = lib['X_valid']
    Y_valid = lib['Y_valid']
    X_valid = X_valid_3D.reshape((X_valid_3D.shape[0], -1))
    Y_valid_oh = one_hot(Y_valid, 10)

    layer_sizes = [256, 256, 10]
    activations = [tf.nn.tanh, tf.nn.tanh, None]
    alpha = 0.01
    iterations = 1000

    save_path = train(X_train, Y_train_oh, X_valid, Y_valid_oh, layer_sizes, activations, alpha, iterations, save_path="./model.ckpt")
    print(f"Model saved in path: {save_path}")
"""
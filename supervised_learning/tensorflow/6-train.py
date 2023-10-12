#!/usr/bin/env python3
import tensorflow.compat.v1 as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, 
          iterations, save_path="/tmp/model.ckpt"):

    """
    Build, train, and save a neural network classifier.

    Arguments:
    - X_train: numpy.ndarray containing the training input data
    - Y_train: numpy.ndarray containing the training labels
    - X_valid: numpy.ndarray containing the validation input data
    - Y_valid: numpy.ndarray containing the validation labels
    - layer_sizes: list containing number of nodes in each layer of the network
    - activations: list containing activation functions
      for each layer of the network
    - alpha: learning rate
    - iterations: number of iterations to train over
    - save_path: path to save the model

    Returns: the path where the model was saved
    """

    # Create placeholders
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])

    # Forward propagation
    y_pred = forward_prop(x, layer_sizes, activations)

    # Calculate accuracy
    accuracy = calculate_accuracy(y, y_pred)

    # Calculate loss
    loss = calculate_loss(y, y_pred)

    # Create training operation
    train_op = create_train_op(loss, alpha)

    # Add to the graph's collection
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

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

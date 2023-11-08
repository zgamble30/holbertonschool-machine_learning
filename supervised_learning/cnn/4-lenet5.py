#!/usr/bin/env python3
"""Modified lenet5."""
import tensorflow.compat.v1 as tf

def lenet5(x, y):
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    # Convolutional Layer 1
    conv1 = tf.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                             activation=tf.nn.relu, kernel_initializer=initializer)(x)

    # Max Pooling Layer 1
    pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    # Convolutional Layer 2
    conv2 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                             activation=tf.nn.relu, kernel_initializer=initializer)(pool1)

    # Max Pooling Layer 2
    pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    # Flatten layer
    flatten = tf.layers.Flatten()(pool2)

    # Fully Connected Layer 1
    fc1 = tf.layers.Dense(units=120, activation=tf.nn.relu, kernel_initializer=initializer)(flatten)

    # Fully Connected Layer 2
    fc2 = tf.layers.Dense(units=84, activation=tf.nn.relu, kernel_initializer=initializer)(fc1)

    # Output Layer
    logits = tf.layers.Dense(units=10, kernel_initializer=initializer)(fc2)

    # Softmax Activation
    y_pred = tf.nn.softmax(logits)

    # Loss function
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)

    # Accuracy calculation
    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    # Training operation using Adam optimizer
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

    return y_pred, train_op, loss, accuracy

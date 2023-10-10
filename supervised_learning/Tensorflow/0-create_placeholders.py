import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """
    Creates two placeholders for a neural network.

    Args:
        nx (int): The number of feature columns in the data.
        classes (int): The number of classes in the classifier.

    Returns:
        x: The placeholder for the input data.
        y: The placeholder for the one-hot labels.
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')

    return x, y

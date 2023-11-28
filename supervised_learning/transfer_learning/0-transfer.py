#!/usr/bin/env python3

import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    Pre-processes the CIFAR-10 data.

    Args:
    X: numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR-10 data,
       where m is the number of data points
    Y: numpy.ndarray of shape (m,) containing the CIFAR-10 labels for X

    Returns: X_preprocessed, Y_preprocessed
        X_preprocessed: numpy.ndarray containing the preprocessed X
        Y_preprocessed: numpy.ndarray containing the preprocessed Y
    """
    X_preprocessed = K.applications.efficientnet_v2.preprocess_input(
        X, data_format="channels_last")
    Y_preprocessed = K.utils.to_categorical(Y, 10)
    return X_preprocessed, Y_preprocessed

if __name__ == '__main__':
    # Load CIFAR-10 dataset
    (X_train_raw, Y_train_raw), (X_test_raw, Y_test_raw) = K.datasets.cifar10.load_data()
    
    # Preprocess the data
    X_train, Y_train = preprocess_data(X_train_raw, Y_train_raw)
    X_test, Y_test = preprocess_data(X_test_raw, Y_test_raw)

    # Define input layer and resize images
    inputs = K.Input(shape=(32, 32, 3))
    inputs_resized = K.layers.Lambda(
        lambda x: K.backend.resize_images(x,
                                          height_factor=(224 // 32),
                                          width_factor=(224 // 32),
                                          data_format="channels_last"))(inputs)

    # Load EfficientNetV2B3 model with pre-trained weights
    efficient_net = K.applications.EfficientNetV2B3(
        include_top=False, weights='imagenet',
        input_shape=(224, 224, 3))

    # Apply EfficientNetV2B3 to resized inputs
    X = efficient_net(inputs_resized, training=False)
    
    # Flatten the output
    X_flatten = K.layers.Flatten()(X)

    # Add custom dense layer for classification
    outputs = K.layers.Dense(10, activation='softmax')(X_flatten)

    # Create the model
    model = K.Model(inputs=inputs, outputs=outputs)

    # Freeze EfficientNetV2B3 layers
    efficient_net.trainable = False

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=K.optimizers.Adam(),
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(x=X_train, y=Y_train,
                        validation_data=(X_test, Y_test),
                        batch_size=600,
                        epochs=7)

    # Save the trained model
    model.save('cifar10.h5')

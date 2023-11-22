#!/usr/bin/env python3
"""
Trains a convolutional neural network to classify CIFAR 10 dataset
"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    Preprocesses the data for the model
    """
    # Preprocess input data using VGG16 preprocessing
    X_p = K.applications.vgg16.preprocess_input(X)
    # One-hot encode the labels
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

if __name__ == '__main__':
    """
    Trains a convolutional neural network on CIFAR 10 dataset
    Saves model to cifar10.h5
    """
    # Load CIFAR-10 dataset
    (X_train_orig, Y_train_orig), (X_test_orig, Y_test_orig) = K.datasets.cifar10.load_data()
    
    # Preprocess the training and testing data
    X_train, Y_train = preprocess_data(X_train_orig, Y_train_orig)
    X_test, Y_test = preprocess_data(X_test_orig, Y_test_orig)

    # Resize images to the input size expected by the chosen application
    input_resized = K.layers.Lambda(
        lambda x: K.backend.resize_images(x,
                                          height_factor=(224 / 32),
                                          width_factor=(224 / 32),
                                          data_format="channels_last"))(X_train)

    # Choose a Keras Application from Keras Applications
    base_model = K.applications.VGG16(weights='imagenet',
                                      include_top=False,
                                      input_tensor=input_resized,
                                      input_shape=(224, 224, 3))

    # Freeze the layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers for classification on top of the base model
    x = K.layers.Flatten()(base_model.output)
    x = K.layers.Dense(256, activation='relu')(x)
    output_layer = K.layers.Dense(10, activation='softmax')(x)

    # Create a new model that includes both the base model and the custom layers
    model = K.models.Model(inputs=base_model.input, outputs=output_layer)

    # Compile the model with the specified loss function, optimizer, and metrics
    model.compile(loss='categorical_crossentropy',
                  optimizer=K.optimizers.Adam(),
                  metrics=['accuracy'])

    # Train the model on the preprocessed data
    history = model.fit(x=X_train, y=Y_train,
                        validation_data=(X_test, Y_test),
                        batch_size=128,
                        epochs=10)

    # Save the trained model
    model.save('cifar10.h5')

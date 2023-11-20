#!/usr/bin/env python3
import tensorflow.keras as K
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10


def preprocess_data(X, Y):
    """Preprocess CIFAR-10 data."""
    X_p = preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

def transfer_model():
    """Build and train the transfer learning model."""
    # Load the pre-trained ResNet50 model without the top (fully connected) layers
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the layers of the pre-trained model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom top layers for CIFAR-10 classification
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Create the transfer learning model
    model = Model(inputs=base_model.input, outputs=x)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Load CIFAR-10 data
    (_, _), (X, Y) = cifar10.load_data()

    # Preprocess the data
    X_p, Y_p = preprocess_data(X, Y)

    # Train the model
    model.fit(X_p, Y_p, epochs=5, batch_size=32, validation_split=0.2)

    # Save the trained model
    model.save('cifar10.h5')

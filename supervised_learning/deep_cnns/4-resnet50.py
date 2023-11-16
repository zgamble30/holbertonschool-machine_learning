#!/usr/bin/env python3
'''ResNet-50 Architecture'''
import tensorflow.keras as K


def resnet50():
   """
   Builds the ResNet-50 architecture as described
   in Deep Residual Learning for Image Recognition (2015).

   Returns:
       Keras model: The ResNet-50 model.
   """
   # Import the identity_block and projection_block functions
   identity_block = __import__('2-identity_block').identity_block
   projection_block = __import__('3-projection_block').projection_block

   # Define the input tensor
   inputs = K.Input(shape=(224, 224, 3))

   # Apply the initial layers
   x = K.layers.Conv2D(64, 7, strides=2, padding='same', kernel_initializer='he_normal')(inputs)
   x = K.layers.BatchNormalization(axis=3)(x)
   x = K.layers.Activation('relu')(x)
   x = K.layers.MaxPool2D(3, strides=2, padding='same')(x)

   # Define the filter sizes for the first set of blocks
   filters = [64, 64, 256]

   # Apply the projection and identity blocks
   x = projection_block(x, filters, s=1)
   x = identity_block(x, filters)
   x = identity_block(x, filters)

   # Define the filter sizes for the second set of blocks
   filters = [128, 128, 512]

   # Apply the projection and identity blocks
   x = projection_block(x, filters)
   x = identity_block(x, filters)
   x = identity_block(x, filters)
   x = identity_block(x, filters)

   # Define the filter sizes for the third set of blocks
   filters = [256, 256, 1024]

   # Apply the projection and identity blocks
   x = projection_block(x, filters)
   for i in range(5):
       x = identity_block(x, filters)

   # Define the filter sizes for the fourth set of blocks
   filters = [512, 512, 2048]

   # Apply the projection and identity blocks
   x = projection_block(x, filters)
   x = identity_block(x, filters)
   x = identity_block(x, filters)

   # Apply the average pooling, flatten, and dense layers
   x = K.layers.AveragePooling2D(pool_size=7, strides=1)(x)
   x = K.layers.Flatten()(x)
   outputs = K.layers.Dense(1000, activation='softmax', kernel_initializer='he_normal')(x)

   # Create the model
   model = K.Model(inputs=inputs, outputs=outputs)

   # Return the model
   return model

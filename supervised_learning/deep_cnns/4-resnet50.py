#!/usr/bin/env python3
'''ResNet-50 Architecture'''
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    '''Builds ResNet-50 model'''
    input_layer = K.Input(shape=(224, 224, 3))
    initializer = K.initializers.he_normal()

    # Initial convolution block
    conv1 = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=2,
        padding='same',
        kernel_initializer=initializer
    )(input_layer)
    batchnorm1 = K.layers.BatchNormalization(axis=3)(conv1)
    relu1 = K.layers.ReLU()(batchnorm1)
    maxpool1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(relu1)

    # Conv2 block
    conv2 = projection_block(maxpool1, [64, 64, 256], s=1)
    id2a = identity_block(conv2, [64, 64, 256])
    id2b = identity_block(id2a, [64, 64, 256])

    # Conv3 block
    conv3 = projection_block(id2b, [128, 128, 512], s=2)
    id3a = identity_block(conv3, [128, 128, 512])
    id3b = identity_block(id3a, [128, 128, 512])
    id3c = identity_block(id3b, [128, 128, 512])

    # Conv4 block
    conv4 = projection_block(id3c, [256, 256, 1024], s=2)
    id4a = identity_block(conv4, [256, 256, 1024])
    id4b = identity_block(id4a, [256, 256, 1024])
    id4c = identity_block(id4b, [256, 256, 1024])
    id4d = identity_block(id4c, [256, 256, 1024])
    id4e = identity_block(id4d, [256, 256, 1024])

    # Conv5 block
    conv5 = projection_block(id4e, [512, 512, 2048], s=2)
    id5a = identity_block(conv5, [512, 512, 2048])
    id5b = identity_block(id5a, [512, 512, 2048])

    # Global average pooling
    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7), strides=1, padding='valid')(id5b)

    # Fully connected layer (Dense)
    dense = K.layers.Dense(units=1000, activation='softmax', kernel_initializer=initializer)(avg_pool)

    # Create and return the model
    model = K.models.Model(inputs=input_layer, outputs=dense, name='ResNet50')
    return model

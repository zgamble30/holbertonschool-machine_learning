#!/usr/bin/env python3
"""builds an inception block """
import tensorflow.keras as K


def inception_block(A_prev, filters):
    F1, F3R, F3, F5R, F5, FPP = filters

    # 1x1 convolution
    conv1x1 = K.layers.Conv2D(filters=F1, kernel_size=(1, 1), padding='same',
                              activation='relu')(A_prev)

    # 1x1 convolution followed by 3x3 convolution
    conv3x3 = K.layers.Conv2D(filters=F3R, kernel_size=(1, 1), padding='same',
                              activation='relu')(A_prev)
    conv3x3 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                              activation='relu')(conv3x3)

    # 1x1 convolution followed by 5x5 convolution
    conv5x5 = K.layers.Conv2D(filters=F5R, kernel_size=(1, 1), padding='same',
                              activation='relu')(A_prev)
    conv5x5 = K.layers.Conv2D(filters=F5, kernel_size=(5, 5), padding='same',
                              activation='relu')(conv5x5)

    # 3x3 max pooling followed by 1x1 convolution
    max_pool_1x1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                                         padding='same')(A_prev)
    max_pool_1x1 = K.layers.Conv2D(filters=FPP, kernel_size=(1, 1),
                                   padding='same', activation='relu')(max_pool_1x1)

    # Concatenate all branches
    inception_block_output = K.layers.concatenate([
        conv1x1, conv3x3, conv5x5, max_pool_1x1
    ])

    return inception_block_output

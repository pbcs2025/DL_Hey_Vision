import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    MaxPooling2D,
    AveragePooling2D
)


def build_max_pool_model():
    """
    Build a model using 2x2 max pooling.
    """

    return Sequential([
        Input(shape=(8, 8, 1)),
        MaxPooling2D(
            pool_size=(2, 2),
            strides=2
        )
    ])


def build_average_pool_model():
    """
    Build a model using 2x2 average pooling.
    """

    return Sequential([
        Input(shape=(8, 8, 1)),
        AveragePooling2D(
            pool_size=(2, 2),
            strides=2
        )
    ])


def create_sample_feature_map():
    """
    Create a simple 8x8 feature map
    for demonstrating pooling.
    """

    return tf.constant([
        [1, 2, 3, 4, 5, 6, 7, 8],
        [2, 3, 4, 5, 6, 7, 8, 9],
        [3, 4, 5, 6, 7, 8, 9, 10],
        [4, 5, 6, 7, 8, 9, 10, 11],
        [5, 6, 7, 8, 9, 10, 11, 12],
        [6, 7, 8, 9, 10, 11, 12, 13],
        [7, 8, 9, 10, 11, 12, 13, 14],
        [8, 9, 10, 11, 12, 13, 14, 15]
    ], dtype=tf.float32)
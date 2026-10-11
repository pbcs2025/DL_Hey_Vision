import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D
)


def build_conv_model(padding, stride):
    """
    Build a simple convolution model
    to demonstrate the effect of
    padding and stride.
    """

    model = Sequential([
        Input(shape=(8, 8, 1)),

        Conv2D(
            filters=1,
            kernel_size=(3, 3),
            strides=stride,
            padding=padding,
            activation=None
        )
    ])

    return model


def calculate_output_size(
    input_size,
    filter_size,
    padding,
    stride
):
    """
    Calculate the spatial output size
    of a convolution layer.
    """

    if padding == "same":
        return (input_size + stride - 1) // stride

    return (
        (input_size - filter_size) // stride
    ) + 1
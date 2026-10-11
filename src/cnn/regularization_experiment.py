import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.regularizers import l2


def build_cnn_baseline():
    """
    CNN without explicit regularization.
    """

    model = Sequential([
        Input(shape=(28, 28, 1)),

        Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        MaxPooling2D((2, 2)),

        Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        MaxPooling2D((2, 2)),

        Flatten(),

        Dense(
            64,
            activation="relu"
        ),

        Dense(
            10,
            activation="softmax"
        )
    ])

    return model


def build_cnn_dropout():
    """
    CNN with Dropout.
    """

    model = Sequential([
        Input(shape=(28, 28, 1)),

        Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        MaxPooling2D((2, 2)),

        Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        MaxPooling2D((2, 2)),

        Flatten(),

        Dense(
            64,
            activation="relu"
        ),

        Dropout(0.5),

        Dense(
            10,
            activation="softmax"
        )
    ])

    return model


def build_cnn_batchnorm():
    """
    CNN with Batch Normalization.
    """

    model = Sequential([
        Input(shape=(28, 28, 1)),

        Conv2D(
            32,
            (3, 3),
            padding="same"
        ),

        BatchNormalization(),

        tf.keras.layers.ReLU(),

        MaxPooling2D((2, 2)),

        Conv2D(
            64,
            (3, 3),
            padding="same"
        ),

        BatchNormalization(),

        tf.keras.layers.ReLU(),

        MaxPooling2D((2, 2)),

        Flatten(),

        Dense(
            64,
            activation="relu"
        ),

        Dense(
            10,
            activation="softmax"
        )
    ])

    return model


def build_cnn_l2():
    """
    CNN with L2 weight regularization.
    """

    model = Sequential([
        Input(shape=(28, 28, 1)),

        Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same",
            kernel_regularizer=l2(0.001)
        ),

        MaxPooling2D((2, 2)),

        Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same",
            kernel_regularizer=l2(0.001)
        ),

        MaxPooling2D((2, 2)),

        Flatten(),

        Dense(
            64,
            activation="relu",
            kernel_regularizer=l2(0.001)
        ),

        Dense(
            10,
            activation="softmax"
        )
    ])

    return model


def compile_model(model):
    """
    Compile a CNN using Adam optimizer.
    """

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
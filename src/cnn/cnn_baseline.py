import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense
)


def build_baseline_cnn():
    """
    Build a simple CNN baseline for MNIST classification.

    Architecture:
        Input
        -> Conv2D
        -> ReLU
        -> MaxPooling
        -> Conv2D
        -> ReLU
        -> MaxPooling
        -> Flatten
        -> Dense
        -> Output
    """

    model = Sequential([
        Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            input_shape=(28, 28, 1)
        ),

        MaxPooling2D(
            pool_size=(2, 2)
        ),

        Conv2D(
            64,
            kernel_size=(3, 3),
            activation="relu",
            padding="same"
        ),

        MaxPooling2D(
            pool_size=(2, 2)
        ),

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

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
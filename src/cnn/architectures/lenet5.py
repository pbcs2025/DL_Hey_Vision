import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    AveragePooling2D,
    Flatten,
    Dense
)


def build_lenet5():
    model = Sequential([
        Input(shape=(28, 28, 1)),

        # C1
        Conv2D(
            filters=6,
            kernel_size=(5, 5),
            activation="tanh",
            padding="valid"
        ),

        # S2
        AveragePooling2D(
            pool_size=(2, 2),
            strides=2
        ),

        # C3
        Conv2D(
            filters=16,
            kernel_size=(5, 5),
            activation="tanh",
            padding="valid"
        ),

        # S4
        AveragePooling2D(
            pool_size=(2, 2),
            strides=2
        ),

        Flatten(),

        # C5 / Fully connected
        Dense(120, activation="tanh"),

        # F6
        Dense(84, activation="tanh"),

        # Output
        Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
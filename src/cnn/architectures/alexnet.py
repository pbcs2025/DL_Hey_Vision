import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)


def build_alexnet():
    model = Sequential([
        Input(shape=(28, 28, 1)),

        # C1
        Conv2D(
            filters=96,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
            activation="relu"
        ),

        # Pool1
        MaxPooling2D(
            pool_size=(2, 2),
            strides=2
        ),

        # C2
        Conv2D(
            filters=256,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),

        # Pool2
        MaxPooling2D(
            pool_size=(2, 2),
            strides=2
        ),

        # C3
        Conv2D(
            filters=384,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),

        # C4
        Conv2D(
            filters=384,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),

        # C5
        Conv2D(
            filters=256,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),

        # Pool5
        MaxPooling2D(
            pool_size=(2, 2),
            strides=2
        ),

        Flatten(),

        # Fully connected layers
        Dense(4096, activation="relu"),
        Dropout(0.5),

        Dense(4096, activation="relu"),
        Dropout(0.5),

        # Output
        Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
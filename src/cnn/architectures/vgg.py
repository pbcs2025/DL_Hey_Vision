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


def build_vgg():
    model = Sequential([
        Input(shape=(28, 28, 1)),

        # Block 1
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),
        MaxPooling2D(
            pool_size=(2, 2),
            strides=2
        ),

        # Block 2
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),
        MaxPooling2D(
            pool_size=(2, 2),
            strides=2
        ),

        # Block 3
        Conv2D(
            filters=256,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),
        Conv2D(
            filters=256,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),
        MaxPooling2D(
            pool_size=(2, 2),
            strides=2
        ),

        # Classification layers
        Flatten(),

        Dense(
            512,
            activation="relu"
        ),
        Dropout(0.5),

        Dense(
            512,
            activation="relu"
        ),
        Dropout(0.5),

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
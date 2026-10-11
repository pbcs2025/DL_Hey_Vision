import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense
)
from tensorflow.keras.optimizers import Adam


def build_model(learning_rate):
    model = Sequential([
        Input(shape=(28, 28, 1)),

        Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu",
            padding="same"
        ),

        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(
            64,
            kernel_size=(3, 3),
            activation="relu",
            padding="same"
        ),

        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        Dense(64, activation="relu"),

        Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def build_decay_model():
    model = Sequential([
        Input(shape=(28, 28, 1)),

        Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu",
            padding="same"
        ),

        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(
            64,
            kernel_size=(3, 3),
            activation="relu",
            padding="same"
        ),

        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        Dense(64, activation="relu"),

        Dense(10, activation="softmax")
    ])

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.9
    )

    optimizer = Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
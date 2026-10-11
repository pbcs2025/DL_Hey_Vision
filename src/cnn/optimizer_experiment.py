import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense
)


def build_cnn():
    """
    Build the same CNN architecture
    for every optimizer experiment.
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


def get_optimizers():
    """
    Return the optimizers used in the comparison.
    """

    return {
        "SGD": tf.keras.optimizers.SGD(
            learning_rate=0.01
        ),

        "Momentum": tf.keras.optimizers.SGD(
            learning_rate=0.01,
            momentum=0.9
        ),

        "RMSProp": tf.keras.optimizers.RMSprop(
            learning_rate=0.001
        ),

        "Adam": tf.keras.optimizers.Adam(
            learning_rate=0.001
        )
    }


def compile_model(model, optimizer):
    """
    Compile the CNN using the selected optimizer.
    """

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
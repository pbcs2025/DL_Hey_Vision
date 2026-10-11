
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input


STATE_SIZE = 6
ACTION_SIZE = 3


def build_dqn():
    model = Sequential([
        Input(shape=(STATE_SIZE,)),

        Dense(64, activation="relu"),
        Dense(64, activation="relu"),

        Dense(ACTION_SIZE, activation="linear")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse"
    )

    return model

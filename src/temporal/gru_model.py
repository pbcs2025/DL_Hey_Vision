from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout


def build_gru_model():
    model = Sequential([
        GRU(
            32,
            input_shape=(30, 4)
        ),
        Dropout(0.2),
        Dense(
            16,
            activation="relu"
        ),
        Dense(
            4,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
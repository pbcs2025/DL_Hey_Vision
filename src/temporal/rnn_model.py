from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout


def build_rnn_model():

    model = Sequential([
        SimpleRNN(32, input_shape=(30, 4)),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(4, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    TextVectorization,
    Embedding,
    LSTM,
    Dense,
    Dropout
)


MAX_TOKENS = 1000
SEQUENCE_LENGTH = 20
EMBEDDING_DIM = 32
NUM_CLASSES = 10


def build_vectorizer():
    return TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH
    )


def build_lstm_intent_model(vectorizer):
    model = Sequential([
        vectorizer,

        Embedding(
            input_dim=MAX_TOKENS,
            output_dim=EMBEDDING_DIM
        ),

        LSTM(32),

        Dropout(0.2),

        Dense(
            32,
            activation="relu"
        ),

        Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
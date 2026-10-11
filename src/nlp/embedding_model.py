import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    TextVectorization,
    Embedding,
    GlobalAveragePooling1D,
    Dense,
    Dropout
)


MAX_TOKENS = 1000
SEQUENCE_LENGTH = 20
EMBEDDING_DIM = 32


def build_vectorizer():
    return TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH
    )


def build_embedding_model(vectorizer, num_classes):
    model = Sequential([
        vectorizer,

        Embedding(
            input_dim=MAX_TOKENS,
            output_dim=EMBEDDING_DIM
        ),

        GlobalAveragePooling1D(),

        Dense(
            32,
            activation="relu"
        ),

        Dropout(0.2),

        Dense(
            num_classes,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
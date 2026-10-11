import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    TextVectorization,
    Embedding,
    MultiHeadAttention,
    LayerNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling1D
)


MAX_TOKENS = 1000
SEQUENCE_LENGTH = 20
EMBEDDING_DIM = 32
NUM_HEADS = 2
NUM_CLASSES = 10


def build_vectorizer():
    return TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH
    )


def build_transformer_intent_model(vectorizer):

    inputs = tf.keras.Input(shape=(), dtype=tf.string)

    x = vectorizer(inputs)

    x = Embedding(
        input_dim=MAX_TOKENS,
        output_dim=EMBEDDING_DIM
    )(x)

    # Self-attention
    attention_output = MultiHeadAttention(
        num_heads=NUM_HEADS,
        key_dim=EMBEDDING_DIM
    )(x, x)

    # Residual connection + normalization
    x = LayerNormalization()(x + attention_output)

    # Feed-forward network
    feed_forward = Dense(
        32,
        activation="relu"
    )(x)

    feed_forward = Dropout(0.2)(feed_forward)

    # Second residual connection + normalization
    x = LayerNormalization()(x + feed_forward)

    # Convert sequence representation into one vector
    x = GlobalAveragePooling1D()(x)

    x = Dense(
        32,
        activation="relu"
    )(x)

    x = Dropout(0.2)(x)

    outputs = Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D
)


def build_autoencoder():

    # Input: 28 x 28 grayscale image
    inputs = Input(shape=(28, 28, 1))

    # --------------------
    # Encoder
    # --------------------

    x = Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    )(inputs)

    x = MaxPooling2D(
        (2, 2),
        padding="same"
    )(x)

    x = Conv2D(
        16,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    encoded = MaxPooling2D(
        (2, 2),
        padding="same"
    )(x)

    # --------------------
    # Decoder
    # --------------------

    x = Conv2D(
        16,
        (3, 3),
        activation="relu",
        padding="same"
    )(encoded)

    x = UpSampling2D(
        (2, 2)
    )(x)

    x = Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    x = UpSampling2D(
        (2, 2)
    )(x)

    # Reconstruct grayscale image
    outputs = Conv2D(
        1,
        (3, 3),
        activation="sigmoid",
        padding="same"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["mae"]
    )

    return model
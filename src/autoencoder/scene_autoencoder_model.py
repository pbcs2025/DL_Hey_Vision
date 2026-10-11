import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D
)


def build_scene_autoencoder():

    # Input: 128 x 128 RGB image
    inputs = Input(shape=(128, 128, 3))

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
        64,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    x = MaxPooling2D(
        (2, 2),
        padding="same"
    )(x)

    x = Conv2D(
        128,
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
        128,
        (3, 3),
        activation="relu",
        padding="same"
    )(encoded)

    x = UpSampling2D(
        (2, 2)
    )(x)

    x = Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

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

    # Reconstruct RGB image
    outputs = Conv2D(
        3,
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
        loss="mse",
        metrics=["mae"]
    )

    return model
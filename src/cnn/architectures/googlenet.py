import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    GlobalAveragePooling2D,
    Dense,
    Concatenate
)


def inception_block(x, filters):
    """
    Simplified Inception block.

    Four parallel branches:
    1. 1x1 convolution
    2. 1x1 convolution + 3x3 convolution
    3. 1x1 convolution + 5x5 convolution
    4. 3x3 max pooling + 1x1 convolution

    The outputs are concatenated along the channel dimension.
    """

    f1, f3_reduce, f3, f5_reduce, f5, f_pool = filters

    # Branch 1: 1x1 convolution
    branch1 = Conv2D(
        f1,
        kernel_size=(1, 1),
        padding="same",
        activation="relu"
    )(x)

    # Branch 2: 1x1 -> 3x3
    branch2 = Conv2D(
        f3_reduce,
        kernel_size=(1, 1),
        padding="same",
        activation="relu"
    )(x)

    branch2 = Conv2D(
        f3,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    )(branch2)

    # Branch 3: 1x1 -> 5x5
    branch3 = Conv2D(
        f5_reduce,
        kernel_size=(1, 1),
        padding="same",
        activation="relu"
    )(x)

    branch3 = Conv2D(
        f5,
        kernel_size=(5, 5),
        padding="same",
        activation="relu"
    )(branch3)

    # Branch 4: max pooling -> 1x1
    branch4 = MaxPooling2D(
        pool_size=(3, 3),
        strides=1,
        padding="same"
    )(x)

    branch4 = Conv2D(
        f_pool,
        kernel_size=(1, 1),
        padding="same",
        activation="relu"
    )(branch4)

    # Concatenate all branches
    return Concatenate(axis=-1)(
        [branch1, branch2, branch3, branch4]
    )


def build_googlenet():
    """
    Simplified GoogLeNet/Inception-style CNN
    adapted for MNIST.
    """

    inputs = Input(shape=(28, 28, 1))

    # Initial feature extraction
    x = Conv2D(
        32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    )(inputs)

    x = MaxPooling2D(
        pool_size=(2, 2),
        strides=2
    )(x)

    # Inception block 1
    x = inception_block(
        x,
        filters=(32, 32, 48, 8, 16, 16)
    )

    # Reduce spatial dimensions
    x = MaxPooling2D(
        pool_size=(2, 2),
        strides=2
    )(x)

    # Inception block 2
    x = inception_block(
        x,
        filters=(64, 48, 64, 12, 24, 24)
    )

    # Global average pooling
    x = GlobalAveragePooling2D()(x)

    # Classification
    outputs = Dense(
        10,
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
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    ReLU,
    Add,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense
)


def residual_block(x, filters, downsample=False):
    """
    A basic ResNet residual block.

    The shortcut connection allows the input to be
    added directly to the output of the convolution path.
    """

    stride = 2 if downsample else 1

    # Main path
    y = Conv2D(
        filters,
        kernel_size=(3, 3),
        strides=stride,
        padding="same",
        use_bias=False
    )(x)

    y = BatchNormalization()(y)
    y = ReLU()(y)

    y = Conv2D(
        filters,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=False
    )(y)

    y = BatchNormalization()(y)

    # Shortcut path
    if downsample or x.shape[-1] != filters:
        x = Conv2D(
            filters,
            kernel_size=(1, 1),
            strides=stride,
            padding="same",
            use_bias=False
        )(x)

        x = BatchNormalization()(x)

    # Residual connection
    y = Add()([x, y])
    y = ReLU()(y)

    return y


def build_resnet():
    """
    ResNet-style CNN adapted for MNIST.
    """

    inputs = Input(shape=(28, 28, 1))

    # Initial convolution
    x = Conv2D(
        32,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False
    )(inputs)

    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Residual blocks
    x = residual_block(x, 32)
    x = residual_block(x, 64, downsample=True)
    x = residual_block(x, 128, downsample=True)

    # Classification head
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
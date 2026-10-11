import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    DepthwiseConv2D,
    BatchNormalization,
    ReLU,
    GlobalAveragePooling2D,
    Dense
)


def depthwise_separable_block(x, filters, stride=1):
    # Depthwise convolution
    x = DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=stride,
        padding="same",
        use_bias=False
    )(x)

    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Pointwise convolution
    x = Conv2D(
        filters,
        kernel_size=(1, 1),
        padding="same",
        use_bias=False
    )(x)

    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x


def build_mobilenet():
    inputs = Input(shape=(28, 28, 1))

    # Initial convolution
    x = Conv2D(
        32,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=False
    )(inputs)

    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Depthwise separable convolution blocks
    x = depthwise_separable_block(x, 32)
    x = depthwise_separable_block(x, 64, stride=2)
    x = depthwise_separable_block(x, 64)
    x = depthwise_separable_block(x, 128, stride=2)
    x = depthwise_separable_block(x, 128)

    # Global average pooling
    x = GlobalAveragePooling2D()(x)

    # Classification layer
    outputs = Dense(
        10,
        activation="softmax"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
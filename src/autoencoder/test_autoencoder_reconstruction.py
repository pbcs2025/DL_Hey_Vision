import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.autoencoder.autoencoder_model import build_autoencoder


MODEL_PATH = "models/autoencoder/conv_autoencoder_bce.keras"


# Load MNIST
(_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

x_test = x_test.astype("float32") / 255.0
x_test = np.expand_dims(x_test, axis=-1)


# Recreate the same noise
rng = np.random.default_rng(42)

noise_factor = 0.30

noise = rng.normal(
    loc=0.0,
    scale=noise_factor,
    size=x_test.shape
)

x_test_noisy = np.clip(
    x_test + noise,
    0.0,
    1.0
)


# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)


# Select samples
sample_images = x_test_noisy[:10]
clean_images = x_test[:10]


# Reconstruct
reconstructed_images = model.predict(
    sample_images,
    verbose=0
)


# Display results
fig, axes = plt.subplots(
    3,
    10,
    figsize=(15, 5)
)

for i in range(10):

    # Noisy
    axes[0, i].imshow(
        sample_images[i].squeeze(),
        cmap="gray"
    )
    axes[0, i].axis("off")

    # Reconstructed
    axes[1, i].imshow(
        reconstructed_images[i].squeeze(),
        cmap="gray"
    )
    axes[1, i].axis("off")

    # Clean
    axes[2, i].imshow(
        clean_images[i].squeeze(),
        cmap="gray"
    )
    axes[2, i].axis("off")


axes[0, 0].set_ylabel("Noisy")
axes[1, 0].set_ylabel("Reconstructed")
axes[2, 0].set_ylabel("Clean")

plt.tight_layout()

plt.savefig(
    "experiments/autoencoder_reconstruction.png",
    dpi=150
)

plt.show()
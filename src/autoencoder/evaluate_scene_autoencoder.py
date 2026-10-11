import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

MODEL_PATH = (
    "models/autoencoder/"
    "scene_autoencoder_mse.keras"
)

SPLIT_DIR = "/kaggle/working/scene_autoencoder_data"

TEST_PATH = os.path.join(
    SPLIT_DIR,
    "test.txt"
)

IMAGE_SIZE = (128, 128)
NOISE_STD = 0.05
SEED = 42

OUTPUT_DIR = "evaluation/autoencoder"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

tf.random.set_seed(SEED)
np.random.seed(SEED)


# --------------------------------------------------
# Load test paths
# --------------------------------------------------

with open(TEST_PATH, "r") as f:
    test_paths = [
        line.strip()
        for line in f
        if line.strip()
    ]

print("\nScene Autoencoder Evaluation")
print("=" * 50)

print("Test images:", len(test_paths))


# --------------------------------------------------
# Load image
# --------------------------------------------------

def load_image(path):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    return image


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

print("\nLoading trained model...")
print("=" * 50)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")
print("Parameters:", model.count_params())


# --------------------------------------------------
# Evaluate test images
# --------------------------------------------------

noisy_mse_values = []
reconstructed_mse_values = []

noisy_mae_values = []
reconstructed_mae_values = []

# Fixed RNG for reproducibility
rng = np.random.default_rng(SEED)


for path in test_paths:

    clean = load_image(path).numpy()

    noise = rng.normal(
        0.0,
        NOISE_STD,
        clean.shape
    )

    noisy = np.clip(
        clean + noise,
        0.0,
        1.0
    )

    reconstructed = model.predict(
        noisy[np.newaxis, ...],
        verbose=0
    )[0]

    # Noisy -> clean
    noisy_mse = np.mean(
        (noisy - clean) ** 2
    )

    noisy_mae = np.mean(
        np.abs(noisy - clean)
    )

    # Reconstruction -> clean
    reconstructed_mse = np.mean(
        (reconstructed - clean) ** 2
    )

    reconstructed_mae = np.mean(
        np.abs(reconstructed - clean)
    )

    noisy_mse_values.append(noisy_mse)
    reconstructed_mse_values.append(
        reconstructed_mse
    )

    noisy_mae_values.append(noisy_mae)
    reconstructed_mae_values.append(
        reconstructed_mae
    )


# --------------------------------------------------
# Calculate averages
# --------------------------------------------------

noisy_mse = np.mean(noisy_mse_values)
reconstructed_mse = np.mean(
    reconstructed_mse_values
)

noisy_mae = np.mean(noisy_mae_values)
reconstructed_mae = np.mean(
    reconstructed_mae_values
)


mse_improvement = (
    (noisy_mse - reconstructed_mse)
    / noisy_mse
) * 100

mae_improvement = (
    (noisy_mae - reconstructed_mae)
    / noisy_mae
) * 100


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\nTest Results")
print("=" * 50)

print(
    f"Noisy Input MSE:       "
    f"{noisy_mse:.6f}"
)

print(
    f"Reconstructed MSE:     "
    f"{reconstructed_mse:.6f}"
)

print(
    f"Noisy Input MAE:       "
    f"{noisy_mae:.6f}"
)

print(
    f"Reconstructed MAE:     "
    f"{reconstructed_mae:.6f}"
)

print(
    f"\nMSE Improvement:       "
    f"{mse_improvement:.2f}%"
)

print(
    f"MAE Improvement:       "
    f"{mae_improvement:.2f}%"
)


# --------------------------------------------------
# Visual comparison
# --------------------------------------------------

print("\nCreating visual comparison...")

visual_count = min(
    5,
    len(test_paths)
)

fig, axes = plt.subplots(
    visual_count,
    3,
    figsize=(12, 4 * visual_count)
)

if visual_count == 1:
    axes = np.expand_dims(
        axes,
        axis=0
    )

# Reset RNG so visual examples
# use the same deterministic noise
visual_rng = np.random.default_rng(SEED)

for row in range(visual_count):

    path = test_paths[row]

    clean = load_image(path).numpy()

    noise = visual_rng.normal(
        0.0,
        NOISE_STD,
        clean.shape
    )

    noisy = np.clip(
        clean + noise,
        0.0,
        1.0
    )

    reconstructed = model.predict(
        noisy[np.newaxis, ...],
        verbose=0
    )[0]

    axes[row, 0].imshow(clean)
    axes[row, 0].set_title(
        "Clean"
    )
    axes[row, 0].axis("off")

    axes[row, 1].imshow(noisy)
    axes[row, 1].set_title(
        "Noisy (σ = 0.05)"
    )
    axes[row, 1].axis("off")

    axes[row, 2].imshow(
        np.clip(
            reconstructed,
            0.0,
            1.0
        )
    )
    axes[row, 2].set_title(
        "Autoencoder Reconstruction"
    )
    axes[row, 2].axis("off")

plt.tight_layout()

visual_path = os.path.join(
    OUTPUT_DIR,
    "scene_autoencoder_comparison.png"
)

plt.savefig(
    visual_path,
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("\nVisual comparison saved to:")
print(visual_path)
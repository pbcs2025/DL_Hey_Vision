import numpy as np
import tensorflow as tf


MSE_MODEL_PATH = "models/autoencoder/conv_autoencoder.keras"
BCE_MODEL_PATH = "models/autoencoder/conv_autoencoder_bce.keras"


# --------------------------------
# Load MNIST test data
# --------------------------------

(_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

x_test = x_test.astype("float32") / 255.0
x_test = np.expand_dims(x_test, axis=-1)


# --------------------------------
# Recreate exactly the same noise
# --------------------------------

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


# --------------------------------
# Load both models
# --------------------------------

mse_model = tf.keras.models.load_model(
    MSE_MODEL_PATH
)

bce_model = tf.keras.models.load_model(
    BCE_MODEL_PATH
)


# --------------------------------
# Generate reconstructions
# --------------------------------

mse_reconstructed = mse_model.predict(
    x_test_noisy,
    batch_size=128,
    verbose=1
)

bce_reconstructed = bce_model.predict(
    x_test_noisy,
    batch_size=128,
    verbose=1
)


# --------------------------------
# Calculate metrics
# --------------------------------

noisy_mse = np.mean(
    (x_test - x_test_noisy) ** 2
)

noisy_mae = np.mean(
    np.abs(x_test - x_test_noisy)
)


mse_autoencoder_mse = np.mean(
    (x_test - mse_reconstructed) ** 2
)

mse_autoencoder_mae = np.mean(
    np.abs(x_test - mse_reconstructed)
)


bce_autoencoder_mse = np.mean(
    (x_test - bce_reconstructed) ** 2
)

bce_autoencoder_mae = np.mean(
    np.abs(x_test - bce_reconstructed)
)


# --------------------------------
# Print results
# --------------------------------

print("\nAutoencoder Comparison")
print("=" * 60)

print("\nFull Test Set")
print("-" * 60)

print("Test images:", len(x_test))
print("Noise factor:", noise_factor)


print("\nReconstruction Metrics")
print("-" * 60)

print(
    f"Noisy Input      | "
    f"MSE: {noisy_mse:.6f} | "
    f"MAE: {noisy_mae:.6f}"
)

print(
    f"MSE Autoencoder  | "
    f"MSE: {mse_autoencoder_mse:.6f} | "
    f"MAE: {mse_autoencoder_mae:.6f}"
)

print(
    f"BCE Autoencoder  | "
    f"MSE: {bce_autoencoder_mse:.6f} | "
    f"MAE: {bce_autoencoder_mae:.6f}"
)


print("\nImprovement Over Noisy Input")
print("-" * 60)

mse_improvement = (
    (noisy_mse - mse_autoencoder_mse)
    / noisy_mse
) * 100

bce_improvement = (
    (noisy_mse - bce_autoencoder_mse)
    / noisy_mse
) * 100


print(
    f"MSE Autoencoder: {mse_improvement:.2f}%"
)

print(
    f"BCE Autoencoder: {bce_improvement:.2f}%"
)
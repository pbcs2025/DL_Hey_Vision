import numpy as np
import tensorflow as tf


MODEL_PATH = "models/autoencoder/conv_autoencoder_bce.keras"


# ----------------------------
# Load MNIST
# ----------------------------

(_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

x_test = x_test.astype("float32") / 255.0
x_test = np.expand_dims(x_test, axis=-1)


# ----------------------------
# Recreate the same noise
# ----------------------------

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


# ----------------------------
# Load model
# ----------------------------

model = tf.keras.models.load_model(MODEL_PATH)


# ----------------------------
# Reconstruct
# ----------------------------

reconstructed = model.predict(
    x_test_noisy[:100],
    verbose=0
)

clean = x_test[:100]
noisy = x_test_noisy[:100]


# ----------------------------
# Calculate MSE
# ----------------------------

reconstruction_mse = np.mean(
    (clean - reconstructed) ** 2
)

noisy_mse = np.mean(
    (clean - noisy) ** 2
)

zero_output_mse = np.mean(
    clean ** 2
)


# ----------------------------
# Print statistics
# ----------------------------

print("\nAutoencoder Diagnostic")
print("=" * 50)

print("\nPixel Statistics")
print("-" * 50)

print(
    f"Clean image mean:          {clean.mean():.6f}"
)

print(
    f"Noisy image mean:          {noisy.mean():.6f}"
)

print(
    f"Reconstructed mean:        {reconstructed.mean():.6f}"
)

print(
    f"Reconstructed min:         {reconstructed.min():.6f}"
)

print(
    f"Reconstructed max:         {reconstructed.max():.6f}"
)


print("\nMSE Comparison")
print("-" * 50)

print(
    f"Noisy → Clean MSE:         {noisy_mse:.6f}"
)

print(
    f"Autoencoder → Clean MSE:   {reconstruction_mse:.6f}"
)

print(
    f"Black output → Clean MSE:  {zero_output_mse:.6f}"
)


print("\nInterpretation")
print("-" * 50)

if reconstruction_mse < noisy_mse:
    print(
        "The autoencoder improves upon the noisy input."
    )
else:
    print(
        "The autoencoder does NOT improve upon the noisy input."
    )

if reconstructed.mean() < 0.05:
    print(
        "WARNING: Reconstruction is strongly biased toward black."
    )
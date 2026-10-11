import numpy as np
import tensorflow as tf

from src.autoencoder.autoencoder_model import build_autoencoder


MODEL_PATH = "models/autoencoder/conv_autoencoder_bce.keras"


# ----------------------------
# Load MNIST
# ----------------------------

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# Add channel dimension
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)


# ----------------------------
# Add Gaussian noise
# ----------------------------

rng = np.random.default_rng(42)

noise_factor = 0.30

train_noise = rng.normal(
    loc=0.0,
    scale=noise_factor,
    size=x_train.shape
)

test_noise = rng.normal(
    loc=0.0,
    scale=noise_factor,
    size=x_test.shape
)

x_train_noisy = x_train + train_noise
x_test_noisy = x_test + test_noise


# Keep pixel values between 0 and 1
x_train_noisy = np.clip(
    x_train_noisy,
    0.0,
    1.0
)

x_test_noisy = np.clip(
    x_test_noisy,
    0.0,
    1.0
)


print("\nAutoencoder Training")
print("=" * 50)

print("Clean training data:", x_train.shape)
print("Noisy training data:", x_train_noisy.shape)

print("Clean testing data:", x_test.shape)
print("Noisy testing data:", x_test_noisy.shape)

print("Noise factor:", noise_factor)


# ----------------------------
# Build model
# ----------------------------

model = build_autoencoder()

print("\nModel Parameters:", model.count_params())


# ----------------------------
# Train
# ----------------------------

history = model.fit(
    x_train_noisy,
    x_train,
    validation_split=0.10,
    epochs=10,
    batch_size=128,
    shuffle=True,
    verbose=1
)


# ----------------------------
# Evaluate
# ----------------------------

test_loss, test_mae = model.evaluate(
    x_test_noisy,
    x_test,
    verbose=0
)

print("\nFinal Evaluation")
print("=" * 50)

print(f"Test MSE: {test_loss:.6f}")
print(f"Test MAE: {test_mae:.6f}")


# ----------------------------
# Save model
# ----------------------------

model.save(MODEL_PATH)

print("\nModel saved to:")
print(MODEL_PATH)
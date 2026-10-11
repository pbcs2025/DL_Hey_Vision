import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint

from .scene_autoencoder_model import build_scene_autoencoder


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SPLIT_DIR = "/kaggle/working/scene_autoencoder_data"

TRAIN_PATH = os.path.join(SPLIT_DIR, "train.txt")
VAL_PATH = os.path.join(SPLIT_DIR, "val.txt")

MODEL_PATH = (
    "models/autoencoder/"
    "scene_autoencoder_mse.keras"
)

IMAGE_SIZE = (128, 128)

NOISE_STD = 0.05

BATCH_SIZE = 32
EPOCHS = 20

SEED = 42


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

tf.random.set_seed(SEED)
np.random.seed(SEED)


# --------------------------------------------------
# Load file paths
# --------------------------------------------------

def load_image_paths(path):
    with open(path, "r") as f:
        paths = [
            line.strip()
            for line in f
            if line.strip()
        ]

    return paths


train_paths = load_image_paths(TRAIN_PATH)
val_paths = load_image_paths(VAL_PATH)


print("\nScene Autoencoder Dataset")
print("=" * 50)

print("Training images:", len(train_paths))
print("Validation images:", len(val_paths))


# --------------------------------------------------
# Image loading
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
# Add Gaussian noise
# --------------------------------------------------

def add_noise(image):

    noise = tf.random.normal(
        shape=tf.shape(image),
        mean=0.0,
        stddev=NOISE_STD
    )

    noisy_image = image + noise

    noisy_image = tf.clip_by_value(
        noisy_image,
        0.0,
        1.0
    )

    return noisy_image


# --------------------------------------------------
# Dataset preprocessing
# --------------------------------------------------

def preprocess(path):

    clean_image = load_image(path)

    noisy_image = add_noise(
        clean_image
    )

    return noisy_image, clean_image


# --------------------------------------------------
# Create tf.data datasets
# --------------------------------------------------

train_dataset = tf.data.Dataset.from_tensor_slices(
    train_paths
)

train_dataset = train_dataset.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_dataset = train_dataset.shuffle(
    buffer_size=1000,
    seed=SEED
)

train_dataset = train_dataset.batch(
    BATCH_SIZE
)

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)


val_dataset = tf.data.Dataset.from_tensor_slices(
    val_paths
)

val_dataset = val_dataset.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_dataset = val_dataset.batch(
    BATCH_SIZE
)

val_dataset = val_dataset.prefetch(
    tf.data.AUTOTUNE
)


# --------------------------------------------------
# Build model
# --------------------------------------------------

print("\nBuilding Scene Autoencoder")
print("=" * 50)

model = build_scene_autoencoder()

model.summary()


# --------------------------------------------------
# Create output directory
# --------------------------------------------------

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)


# --------------------------------------------------
# Save best model
# --------------------------------------------------

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)


# --------------------------------------------------
# Train
# --------------------------------------------------

print("\nStarting training...")
print("=" * 50)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)


# --------------------------------------------------
# Final evaluation
# --------------------------------------------------

print("\nFinal Training Results")
print("=" * 50)

best_val_loss = min(
    history.history["val_loss"]
)

final_train_loss = (
    history.history["loss"][-1]
)

final_val_loss = (
    history.history["val_loss"][-1]
)

print(
    f"Best validation loss: "
    f"{best_val_loss:.6f}"
)

print(
    f"Final training loss: "
    f"{final_train_loss:.6f}"
)

print(
    f"Final validation loss: "
    f"{final_val_loss:.6f}"
)

print("\nBest model saved to:")
print(MODEL_PATH)
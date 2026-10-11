import os
import random
import numpy as np
import tensorflow as tf


# --------------------------------------------------
# Configuration
# --------------------------------------------------

COCO_IMAGE_DIR = (
    "/kaggle/input/datasets/awsaf49/"
    "coco-2017-dataset/coco2017/train2017"
)

OUTPUT_DIR = "/kaggle/working/scene_autoencoder_data"

TRAIN_COUNT = 5000
VAL_COUNT = 1000
TEST_COUNT = 500

IMAGE_SIZE = (128, 128)

SEED = 42


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)


# --------------------------------------------------
# Find COCO images
# --------------------------------------------------

print("\nFinding COCO images...")
print("=" * 50)

image_files = [
    os.path.join(COCO_IMAGE_DIR, filename)
    for filename in os.listdir(COCO_IMAGE_DIR)
    if filename.lower().endswith(".jpg")
]

print("Available images:", len(image_files))


# --------------------------------------------------
# Check dataset size
# --------------------------------------------------

required_images = (
    TRAIN_COUNT +
    VAL_COUNT +
    TEST_COUNT
)

if len(image_files) < required_images:
    raise ValueError(
        f"Not enough images. "
        f"Required: {required_images}, "
        f"Available: {len(image_files)}"
    )


# --------------------------------------------------
# Fixed random selection
# --------------------------------------------------

rng = random.Random(SEED)

selected_images = rng.sample(
    image_files,
    required_images
)

train_files = selected_images[
    :TRAIN_COUNT
]

val_files = selected_images[
    TRAIN_COUNT:
    TRAIN_COUNT + VAL_COUNT
]

test_files = selected_images[
    TRAIN_COUNT + VAL_COUNT:
]


# --------------------------------------------------
# Save split lists
# --------------------------------------------------

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

with open(
    os.path.join(OUTPUT_DIR, "train.txt"),
    "w"
) as f:
    f.write("\n".join(train_files))

with open(
    os.path.join(OUTPUT_DIR, "val.txt"),
    "w"
) as f:
    f.write("\n".join(val_files))

with open(
    os.path.join(OUTPUT_DIR, "test.txt"),
    "w"
) as f:
    f.write("\n".join(test_files))


# --------------------------------------------------
# Print split information
# --------------------------------------------------

print("\nDataset Split")
print("=" * 50)

print("Training images:", len(train_files))
print("Validation images:", len(val_files))
print("Testing images:", len(test_files))

print("\nRandom seed:", SEED)

print("\nSaved split files to:")
print(OUTPUT_DIR)   
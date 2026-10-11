import tensorflow as tf

from src.cnn.cnn_baseline import build_baseline_cnn


print("\nAI Vision Assistant - CNN Baseline Experiment")
print("=" * 60)


# --------------------------------------------------
# 1. Load MNIST dataset
# --------------------------------------------------

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# --------------------------------------------------
# 2. Normalize pixel values
# --------------------------------------------------

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# --------------------------------------------------
# 3. Add channel dimension
# --------------------------------------------------

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]


print("\nDataset")
print("-" * 60)
print("Training images:", x_train.shape)
print("Testing images :", x_test.shape)


# --------------------------------------------------
# 4. Build CNN
# --------------------------------------------------

model = build_baseline_cnn()


print("\nCNN Architecture")
print("-" * 60)

model.summary()


# --------------------------------------------------
# 5. Train CNN
# --------------------------------------------------

print("\nTraining CNN...")
print("-" * 60)

history = model.fit(
    x_train,
    y_train,
    validation_split=0.10,
    epochs=5,
    batch_size=128,
    verbose=1
)


# --------------------------------------------------
# 6. Evaluate CNN
# --------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)


print("\nFinal Evaluation")
print("-" * 60)
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")


# --------------------------------------------------
# 7. Save model
# --------------------------------------------------

MODEL_PATH = "models/cnn/baseline_cnn.keras"

model.save(MODEL_PATH)

print("\nModel saved to:")
print(MODEL_PATH)

print("\nCNN baseline experiment completed.")
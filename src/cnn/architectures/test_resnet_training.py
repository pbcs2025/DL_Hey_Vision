import tensorflow as tf

from src.cnn.architectures.resnet import build_resnet


# --------------------------------------------------
# Load MNIST
# --------------------------------------------------

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]


# --------------------------------------------------
# Create validation set
# --------------------------------------------------

validation_size = 6000

x_val = x_train[-validation_size:]
y_val = y_train[-validation_size:]

x_train = x_train[:-validation_size]
y_train = y_train[:-validation_size]


# --------------------------------------------------
# Build ResNet
# --------------------------------------------------

model = build_resnet()

print("\nResNet-Style Training Experiment")
print("=" * 60)

print("Training data:", x_train.shape)
print("Validation data:", x_val.shape)
print("Testing data:", x_test.shape)

print("\nModel Parameters")
print("-" * 60)
print("Total parameters:", model.count_params())


# --------------------------------------------------
# Train
# --------------------------------------------------

print("\nTraining ResNet-Style Model")
print("-" * 60)

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=5,
    batch_size=128,
    verbose=1
)


# --------------------------------------------------
# Evaluate
# --------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)


print("\nFinal Results")
print("=" * 60)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")


# --------------------------------------------------
# Save model
# --------------------------------------------------

MODEL_PATH = "models/cnn/resnet_style.keras"

model.save(MODEL_PATH)

print("\nModel saved to:")
print(MODEL_PATH)
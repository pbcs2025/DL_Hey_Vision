import tensorflow as tf

from src.cnn.lr_decay_experiment import (
    build_model,
    build_decay_model
)


# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Use the same validation split for both models
validation_size = 6000

x_val = x_train[-validation_size:]
y_val = y_train[-validation_size:]

x_train = x_train[:-validation_size]
y_train = y_train[:-validation_size]


print("\nLearning Rate Decay Experiment")
print("=" * 60)

print("Training data:", x_train.shape)
print("Validation data:", x_val.shape)
print("Testing data:", x_test.shape)


# --------------------------------------------------
# Model 1: Adam with constant learning rate
# --------------------------------------------------

print("\nTraining Model 1: Adam - Constant Learning Rate")
print("-" * 60)

constant_model = build_model(learning_rate=0.001)

constant_history = constant_model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=5,
    batch_size=128,
    verbose=1
)

constant_loss, constant_accuracy = constant_model.evaluate(
    x_test,
    y_test,
    verbose=0
)


# --------------------------------------------------
# Model 2: Adam with exponential learning rate decay
# --------------------------------------------------

print("\nTraining Model 2: Adam - Learning Rate Decay")
print("-" * 60)

decay_model = build_decay_model()

decay_history = decay_model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=5,
    batch_size=128,
    verbose=1
)

decay_loss, decay_accuracy = decay_model.evaluate(
    x_test,
    y_test,
    verbose=0
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\nFinal Results")
print("=" * 60)

print(
    f"Adam - Constant LR   | "
    f"Test Accuracy: {constant_accuracy:.4f} | "
    f"Test Loss: {constant_loss:.4f}"
)

print(
    f"Adam - LR Decay      | "
    f"Test Accuracy: {decay_accuracy:.4f} | "
    f"Test Loss: {decay_loss:.4f}"
)


# --------------------------------------------------
# Compare results
# --------------------------------------------------

accuracy_difference = decay_accuracy - constant_accuracy

print("\nComparison")
print("=" * 60)

print(
    f"Accuracy difference (Decay - Constant): "
    f"{accuracy_difference:+.4f}"
)

if accuracy_difference > 0:
    print("Learning rate decay improved test accuracy.")
elif accuracy_difference < 0:
    print("Constant learning rate performed better.")
else:
    print("Both models achieved the same test accuracy.")


# --------------------------------------------------
# Save models
# --------------------------------------------------

constant_model.save(
    "models/cnn/adam_constant_lr.keras"
)

decay_model.save(
    "models/cnn/adam_lr_decay.keras"
)

print("\nModels saved:")
print("models/cnn/adam_constant_lr.keras")
print("models/cnn/adam_lr_decay.keras")
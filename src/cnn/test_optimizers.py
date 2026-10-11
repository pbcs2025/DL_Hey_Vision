import tensorflow as tf

from src.cnn.optimizer_experiment import (
    build_cnn,
    get_optimizers,
    compile_model
)


print("\nAI Vision Assistant - CNN Optimizer Experiment")
print("=" * 70)


# --------------------------------------------------
# 1. Load MNIST
# --------------------------------------------------

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)


# --------------------------------------------------
# 2. Normalize
# --------------------------------------------------

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# --------------------------------------------------
# 3. Add channel dimension
# --------------------------------------------------

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]


print("\nDataset")
print("-" * 70)
print("Training:", x_train.shape)
print("Testing :", x_test.shape)


# --------------------------------------------------
# 4. Get optimizers
# --------------------------------------------------

optimizers = get_optimizers()

results = {}


# --------------------------------------------------
# 5. Train each optimizer
# --------------------------------------------------

for name, optimizer in optimizers.items():

    print("\n")
    print("=" * 70)
    print(f"Training with: {name}")
    print("=" * 70)

    # Create a fresh model for a fair comparison
    model = build_cnn()

    compile_model(
        model,
        optimizer
    )

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.10,
        epochs=5,
        batch_size=128,
        verbose=1
    )

    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    results[name] = {
        "train_accuracy":
            history.history["accuracy"][-1],

        "val_accuracy":
            history.history["val_accuracy"][-1],

        "test_accuracy":
            test_accuracy,

        "test_loss":
            test_loss
    }

    print(f"\n{name} Results")
    print("-" * 70)
    print(
        f"Train Accuracy : "
        f"{history.history['accuracy'][-1]:.4f}"
    )
    print(
        f"Val Accuracy   : "
        f"{history.history['val_accuracy'][-1]:.4f}"
    )
    print(
        f"Test Accuracy  : "
        f"{test_accuracy:.4f}"
    )
    print(
        f"Test Loss      : "
        f"{test_loss:.4f}"
    )


# --------------------------------------------------
# 6. Final comparison
# --------------------------------------------------

print("\n\nFinal Optimizer Comparison")
print("=" * 70)

print(
    f"{'Optimizer':<15}"
    f"{'Train Acc':<15}"
    f"{'Val Acc':<15}"
    f"{'Test Acc':<15}"
    f"{'Test Loss':<15}"
)

print("-" * 70)

for name, result in results.items():

    print(
        f"{name:<15}"
        f"{result['train_accuracy']:<15.4f}"
        f"{result['val_accuracy']:<15.4f}"
        f"{result['test_accuracy']:<15.4f}"
        f"{result['test_loss']:<15.4f}"
    )


print("\nOptimizer experiment completed.")
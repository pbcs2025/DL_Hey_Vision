import tensorflow as tf

from src.cnn.regularization_experiment import (
    build_cnn_baseline,
    build_cnn_dropout,
    build_cnn_batchnorm,
    build_cnn_l2,
    compile_model
)


print("\nAI Vision Assistant - CNN Regularization Experiment")
print("=" * 65)


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
print("-" * 65)
print("Training:", x_train.shape)
print("Testing :", x_test.shape)


# --------------------------------------------------
# 4. Models
# --------------------------------------------------

models = {
    "Baseline": build_cnn_baseline(),
    "Dropout": build_cnn_dropout(),
    "BatchNorm": build_cnn_batchnorm(),
    "L2": build_cnn_l2()
}


# --------------------------------------------------
# 5. Train and evaluate
# --------------------------------------------------

results = {}


for name, model in models.items():

    print("\n")
    print("=" * 65)
    print(f"Training: {name}")
    print("=" * 65)

    compile_model(model)

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
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "final_train_accuracy": history.history["accuracy"][-1],
        "final_val_accuracy": history.history["val_accuracy"][-1]
    }

    print(f"\n{name} Results")
    print("-" * 65)
    print(f"Train Accuracy : {history.history['accuracy'][-1]:.4f}")
    print(f"Val Accuracy   : {history.history['val_accuracy'][-1]:.4f}")
    print(f"Test Accuracy  : {test_accuracy:.4f}")
    print(f"Test Loss      : {test_loss:.4f}")


# --------------------------------------------------
# 6. Comparison
# --------------------------------------------------

print("\n\nFinal Comparison")
print("=" * 65)

print(
    f"{'Model':<15}"
    f"{'Train Acc':<15}"
    f"{'Val Acc':<15}"
    f"{'Test Acc':<15}"
    f"{'Test Loss':<15}"
)

print("-" * 65)

for name, result in results.items():

    print(
        f"{name:<15}"
        f"{result['final_train_accuracy']:<15.4f}"
        f"{result['final_val_accuracy']:<15.4f}"
        f"{result['test_accuracy']:<15.4f}"
        f"{result['test_loss']:<15.4f}"
    )


print("\nRegularization experiment completed.")
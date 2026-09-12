import numpy as np
import os
from sklearn.model_selection import train_test_split

from .create_temporal_dataset import create_dataset
from .lstm_model import build_lstm_model


# Create dataset
X, y = create_dataset(250)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# Build LSTM
model = build_lstm_model()


# Train
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


# Evaluate
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nFinal Results")
print("=" * 40)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

os.makedirs("models/temporal", exist_ok=True)

model.save("models/temporal/approaching_lstm.keras")

print("\nModel saved")
print("="*40)
print("Path: models/temporal/approaching_lstm.keras")
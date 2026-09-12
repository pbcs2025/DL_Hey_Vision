import numpy as np
from sklearn.model_selection import train_test_split

from .risk_dataset import create_risk_dataset
from .risk_model import build_risk_model


# --------------------------------------------------
# 1. Create the dataset
# --------------------------------------------------

X, y = create_risk_dataset(samples_per_class=500)

print("\nRisk Dataset")
print("=" * 40)
print("Features:", X.shape)
print("Labels:", y.shape)


# --------------------------------------------------
# 2. Split into training and testing data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# --------------------------------------------------
# 3. Build the ANN
# --------------------------------------------------

model = build_risk_model()

print("\nANN Architecture")
print("=" * 40)

model.summary()


# --------------------------------------------------
# 4. Train the ANN
# --------------------------------------------------

print("\nTraining ANN...")
print("=" * 40)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=20,
    batch_size=32,
    verbose=1
)


# --------------------------------------------------
# 5. Evaluate the ANN
# --------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nFinal Evaluation")
print("=" * 40)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


# --------------------------------------------------
# 6. Save the trained model
# --------------------------------------------------

MODEL_PATH = "models/risk/risk_ann.keras"

model.save(MODEL_PATH)

print("\nModel saved to:")
print(MODEL_PATH)
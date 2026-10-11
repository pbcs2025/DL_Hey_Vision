import numpy as np

from src.temporal.bilstm_model import build_bilstm_model


# --------------------------------------------------
# Generate synthetic temporal data
# --------------------------------------------------

np.random.seed(42)

NUM_SAMPLES = 1000
SEQUENCE_LENGTH = 30
NUM_FEATURES = 4
NUM_CLASSES = 4

X = []
y = []


for _ in range(NUM_SAMPLES):

    movement = np.random.randint(0, NUM_CLASSES)

    sequence = []

    x_position = np.random.uniform(0.2, 0.8)
    y_position = np.random.uniform(0.2, 0.8)
    width = np.random.uniform(0.1, 0.3)
    height = np.random.uniform(0.1, 0.3)

    for t in range(SEQUENCE_LENGTH):

        if movement == 0:
            # Stationary
            x = x_position + np.random.normal(0, 0.005)
            y_pos = y_position + np.random.normal(0, 0.005)

        elif movement == 1:
            # Approaching
            x = x_position + np.random.normal(0, 0.005)
            y_pos = y_position + np.random.normal(0, 0.005)

            width += 0.006
            height += 0.006

        elif movement == 2:
            # Moving away
            x = x_position + np.random.normal(0, 0.005)
            y_pos = y_position + np.random.normal(0, 0.005)

            width -= 0.004
            height -= 0.004

        else:
            # Crossing
            x = x_position + (t * 0.015) + np.random.normal(0, 0.005)
            y_pos = y_position + np.random.normal(0, 0.005)

        sequence.append([
            x,
            y_pos,
            width,
            height
        ])

    X.append(sequence)
    y.append(movement)


X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)


# --------------------------------------------------
# Train / test split
# --------------------------------------------------

X_train = X[:800]
y_train = y[:800]

X_test = X[800:]
y_test = y[800:]


print("\nBiLSTM Temporal Training Experiment")
print("=" * 60)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# --------------------------------------------------
# Build model
# --------------------------------------------------

model = build_bilstm_model()

print("\nModel Parameters")
print("-" * 60)
print("Total parameters:", model.count_params())


# --------------------------------------------------
# Train
# --------------------------------------------------

print("\nTraining BiLSTM")
print("-" * 60)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=10,
    batch_size=32,
    verbose=1
)


# --------------------------------------------------
# Evaluate
# --------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)


print("\nFinal Evaluation")
print("=" * 60)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


# --------------------------------------------------
# Save model
# --------------------------------------------------

MODEL_PATH = "models/temporal/bilstm_model.keras"

model.save(MODEL_PATH)

print("\nModel saved to:")
print(MODEL_PATH)
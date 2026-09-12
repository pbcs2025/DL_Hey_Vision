import numpy as np
from tensorflow.keras.models import load_model

from .create_temporal_dataset import (
    generate_stationary_sequence,
    generate_approaching_sequence,
    generate_moving_away_sequence,
    generate_crossing_sequence
)

MODEL_PATH = "models/temporal/approaching_lstm.keras"

CLASS_NAMES = [
    "Stationary",
    "Approaching",
    "Moving Away",
    "Crossing"
]

model = load_model(MODEL_PATH)

test_sequences = [
    ("Stationary", generate_stationary_sequence()),
    ("Approaching", generate_approaching_sequence()),
    ("Moving Away", generate_moving_away_sequence()),
    ("Crossing", generate_crossing_sequence())
]

print("\nLSTM Movement Classification Test")
print("=" * 50)

for actual_class, sequence in test_sequences:

    X = np.array([sequence], dtype=np.float32)

    prediction = model.predict(X, verbose=0)

    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]

    print(f"\nActual:     {actual_class}")
    print(f"Predicted:  {CLASS_NAMES[predicted_class]}")
    print(f"Confidence: {confidence:.4f}")
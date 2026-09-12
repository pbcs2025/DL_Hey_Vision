import numpy as np
from tensorflow.keras.models import load_model

from .risk_dataset import RISK_CLASSES


MODEL_PATH = "models/risk/risk_ann.keras"


# --------------------------------------------------
# Load trained ANN
# --------------------------------------------------

model = load_model(MODEL_PATH)


# --------------------------------------------------
# Helper function
# --------------------------------------------------

def predict_risk(
    object_type,
    position,
    size,
    confidence,
    movement,
    ahead
):
    """
    Predict risk level for one detected object.

    Encodings:

    Position:
        left   = 0
        center = 1
        right  = 2

    Movement:
        stationary   = 0
        approaching  = 1
        moving away  = 2
        crossing     = 3
    """

    position_map = {
        "left": 0,
        "center": 1,
        "right": 2
    }

    movement_map = {
        "stationary": 0,
        "approaching": 1,
        "moving away": 2,
        "crossing": 3
    }

    features = np.array([[
        object_type / 9.0,
        position_map[position] / 2.0,
        size,
        confidence,
        movement_map[movement] / 3.0,
        ahead
    ]], dtype=np.float32)

    prediction = model.predict(
        features,
        verbose=0
    )

    predicted_class = np.argmax(prediction[0])
    confidence_score = prediction[0][predicted_class]

    return (
        RISK_CLASSES[predicted_class],
        confidence_score
    )


# --------------------------------------------------
# Test cases
# --------------------------------------------------

test_cases = [
    {
        "name": "Small stationary chair on left",
        "object_type": 0,
        "position": "left",
        "size": 0.10,
        "confidence": 0.80,
        "movement": "stationary",
        "ahead": 0
    },

    {
        "name": "Moderate chair in center",
        "object_type": 0,
        "position": "center",
        "size": 0.35,
        "confidence": 0.85,
        "movement": "stationary",
        "ahead": 1
    },

    {
        "name": "Large table directly ahead",
        "object_type": 1,
        "position": "center",
        "size": 0.65,
        "confidence": 0.90,
        "movement": "stationary",
        "ahead": 1
    },

    {
        "name": "Approaching person directly ahead",
        "object_type": 2,
        "position": "center",
        "size": 0.70,
        "confidence": 0.95,
        "movement": "approaching",
        "ahead": 1
    },

    {
        "name": "Person crossing from right",
        "object_type": 2,
        "position": "right",
        "size": 0.55,
        "confidence": 0.92,
        "movement": "crossing",
        "ahead": 1
    }
]


# --------------------------------------------------
# Run tests
# --------------------------------------------------

print("\nAI Vision Assistant - ANN Risk Test")
print("=" * 60)

for case in test_cases:

    risk, confidence = predict_risk(
        case["object_type"],
        case["position"],
        case["size"],
        case["confidence"],
        case["movement"],
        case["ahead"]
    )

    print(f"\nCase: {case['name']}")
    print(f"Position: {case['position']}")
    print(f"Size: {case['size']:.2f}")
    print(f"Movement: {case['movement']}")
    print(f"Ahead: {case['ahead']}")
    print(f"Predicted Risk: {risk}")
    print(f"Confidence: {confidence:.4f}")
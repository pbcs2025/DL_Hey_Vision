import numpy as np


RISK_CLASSES = [
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL"
]


# Object classes used by the YOLO detector
OBJECT_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "backpack",
    "bottle",
    "chair",
    "table",
    "traffic_light",
    "stop_sign",
    "door"
]


def create_risk_dataset(samples_per_class=500):

    X = []
    y = []

    rng = np.random.default_rng(42)

    for risk_level in range(4):

        for _ in range(samples_per_class):

            # Object type: 0-12
            object_type = rng.integers(0, len(OBJECT_CLASSES))

            if risk_level == 0:

                position = rng.choice([0, 2])
                size = rng.uniform(0.02, 0.25)
                confidence = rng.uniform(0.50, 0.85)
                movement = rng.choice([0, 2])
                ahead = 0

            elif risk_level == 1:

                position = rng.integers(0, 3)
                size = rng.uniform(0.15, 0.45)
                confidence = rng.uniform(0.55, 0.95)
                movement = rng.choice([0, 2, 3])
                ahead = rng.choice([0, 1])

            elif risk_level == 2:

                position = rng.integers(0, 3)
                size = rng.uniform(0.35, 0.70)
                confidence = rng.uniform(0.65, 1.00)
                movement = rng.choice([1, 3])
                ahead = rng.choice([0, 1])

            else:

                position = 1
                size = rng.uniform(0.55, 0.80)
                confidence = rng.uniform(0.75, 1.00)
                movement = rng.choice([1, 3])
                ahead = 1

            features = [
                object_type / 12.0,
                position / 2.0,
                size,
                confidence,
                movement / 3.0,
                ahead
            ]

            X.append(features)
            y.append(risk_level)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    return X, y
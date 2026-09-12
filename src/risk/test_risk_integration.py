import numpy as np

from ultralytics import YOLO
from tensorflow.keras.models import load_model

from src.reasoning.spatial_reasoning import process_yolo_detections
from src.reasoning.obstacle_reasoning import find_obstacles_ahead
from src.risk.risk_features import create_risk_features
from src.response.risk_response import generate_risk_response


YOLO_MODEL_PATH = "backup/Experiment_02_Backup/best.pt"
RISK_MODEL_PATH = "models/risk/risk_ann.keras"
IMAGE_PATH = "data/test/test_scene.jpg"


# --------------------------------------------------
# Load models
# --------------------------------------------------

yolo_model = YOLO(YOLO_MODEL_PATH)
risk_model = load_model(RISK_MODEL_PATH)


# --------------------------------------------------
# Run YOLO object detection
# --------------------------------------------------

results = yolo_model(
    IMAGE_PATH,
    conf=0.25,
    verbose=False
)


# Convert YOLO detections into our format
detections = process_yolo_detections(results)


# --------------------------------------------------
# Find potential obstacles ahead
# --------------------------------------------------

obstacles = find_obstacles_ahead(detections)


# Image dimensions
image_height, image_width = results[0].orig_shape


print("\nAI Vision Assistant - Query 6")
print("=" * 60)

print("\nPotential Obstacles Ahead")
print("-" * 60)


if not obstacles:

    print("No potential obstacles detected ahead.")

else:

    for obstacle in obstacles:

        object_name = obstacle["object"]
        position = obstacle["position"]
        confidence = obstacle["confidence"]

        # --------------------------------------------------
        # Create ANN risk features
        # --------------------------------------------------

        features = create_risk_features(
            object_name=object_name,
            position=position,
            width=obstacle["width"],
            height=obstacle["height"],
            image_width=image_width,
            image_height=image_height,
            confidence=confidence,
            movement="stationary",
            ahead=True
        )


        # --------------------------------------------------
        # Predict risk using ANN
        # --------------------------------------------------

        prediction = risk_model.predict(
            np.array([features], dtype=np.float32),
            verbose=0
        )[0]


        risk_index = prediction.argmax()

        risk_levels = [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
        ]

        risk_level = risk_levels[risk_index]
        risk_confidence = prediction[risk_index]


        # --------------------------------------------------
        # Generate assistant response
        # --------------------------------------------------

        response = generate_risk_response(
            object_name,
            "directly ahead",
            risk_level,
            risk_confidence
        )


        # --------------------------------------------------
        # Display result
        # --------------------------------------------------

        print(f"\nObject: {object_name}")
        print(f"Position: {position}")
        print(f"Detection confidence: {confidence:.4f}")
        print(f"Risk level: {risk_level}")
        print(f"Risk confidence: {risk_confidence:.4f}")
        print(f"Assistant: {response}")


print("\n")
print("Risk integration test completed.")
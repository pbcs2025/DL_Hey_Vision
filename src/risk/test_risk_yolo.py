import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

from .risk_features import create_risk_features
from .risk_dataset import RISK_CLASSES


# --------------------------------------------------
# Paths
# --------------------------------------------------

YOLO_MODEL_PATH = "backup/Experiment_02_Backup/best.pt"
RISK_MODEL_PATH = "models/risk/risk_ann.keras"
IMAGE_PATH = "data/test/test_scene.jpg"


# --------------------------------------------------
# Load models
# --------------------------------------------------

yolo_model = YOLO(YOLO_MODEL_PATH)
risk_model = load_model(RISK_MODEL_PATH)


# --------------------------------------------------
# Run YOLO
# --------------------------------------------------

results = yolo_model.predict(
    source=IMAGE_PATH,
    conf=0.25,
    verbose=False
)

result = results[0]


# --------------------------------------------------
# Process detections
# --------------------------------------------------

print("\nAI Vision Assistant - YOLO + ANN Risk Test")
print("=" * 60)

if result.boxes is None:
    print("\nNo objects detected.")
    raise SystemExit


image_height, image_width = result.orig_shape

print(f"\nImage size: {image_width} x {image_height}")

print("\nRisk Predictions")
print("-" * 60)


for index, box in enumerate(result.boxes):

    x1, y1, x2, y2 = box.xyxy[0].tolist()

    class_id = int(box.cls[0])
    detection_confidence = float(box.conf[0])

    object_name = result.names[class_id]

    # ----------------------------------------------
    # Calculate position
    # ----------------------------------------------

    x_center = (x1 + x2) / 2

    normalized_x = x_center / image_width

    if normalized_x < 0.33:
        position = "left"
    elif normalized_x < 0.66:
        position = "center"
    else:
        position = "right"

    # ----------------------------------------------
    # Check whether object is a potential obstacle
    # ----------------------------------------------

    obstacle_objects = [
        "chair",
        "table",
        "person",
        "car",
        "bicycle",
        "motorcycle",
        "bus",
        "truck",
        "backpack",
        "door"
    ]

    if object_name not in obstacle_objects:
        continue

    if position != "center":
        continue

    # ----------------------------------------------
    # For Query 6, objects are initially treated
    # as stationary because this is a single image.
    # ----------------------------------------------

    movement = "stationary"

    ahead = True

    # ----------------------------------------------
    # Create ANN features
    # ----------------------------------------------

    features = create_risk_features(
        object_name=object_name,
        position=position,
        width=x2 - x1,
        height=y2 - y1,
        image_width=image_width,
        image_height=image_height,
        confidence=detection_confidence,
        movement=movement,
        ahead=ahead
    )

    X = np.array(
        [features],
        dtype=np.float32
    )

    # ----------------------------------------------
    # Predict risk
    # ----------------------------------------------

    prediction = risk_model.predict(
        X,
        verbose=0
    )

    predicted_class = np.argmax(prediction[0])

    risk = RISK_CLASSES[predicted_class]

    risk_confidence = prediction[0][predicted_class]

    # ----------------------------------------------
    # Display
    # ----------------------------------------------

    print(f"\nObject: {object_name}")
    print(f"Position: {position}")
    print(f"Detection confidence: {detection_confidence:.4f}")
    print(f"Normalized size: {features[2]:.4f}")
    print(f"Risk: {risk}")
    print(f"Risk confidence: {risk_confidence:.4f}")


print("\nRisk integration test completed.")
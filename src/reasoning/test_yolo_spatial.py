from pathlib import Path
from ultralytics import YOLO

from .spatial_reasoning import process_yolo_detections
from ..response.scene_summary import summarize_scene
from .directional_reasoning import get_front_objects
from ..response.front_objects_response import generate_front_objects_response
from .object_location import get_object_location
from ..response.object_location_response import generate_object_location_response


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Paths
MODEL_PATH = PROJECT_ROOT / "backup" / "Experiment_02_Backup" / "best.pt"
IMAGE_PATH = PROJECT_ROOT / "data" / "test" / "test_scene.jpg"


# Load our trained YOLO11n model
model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")
print("Classes:", model.names)


# Run object detection
results = model(
    str(IMAGE_PATH),
    conf=0.25
)


# Convert YOLO detections into spatial information
detections = process_yolo_detections(results)


# Display results
print("\nSpatial Detections")
print("=" * 50)

if not detections:
    print("No objects detected.")

else:
    for detection in detections:
        print(
            f"{detection['object']} → "
            f"{detection['position']} "
            f"(confidence: {detection['confidence']:.2f})"
        )


# Find the location of the door
door_locations = get_object_location(
    detections,
    "door"
)

print("\nDoor Location")
print("=" * 50)

if not door_locations:
    print("Door not detected.")

else:
    for location in door_locations:
        print(f"Door → {location}")


# Generate a natural-language response for the door
door_response = generate_object_location_response(
    "door",
    door_locations
)

print("\nDoor Location Response")
print("=" * 50)
print(door_response)

# Find objects in the central/front region
front_objects = get_front_objects(detections)

print("\nObjects In Front")
print("=" * 50)

if not front_objects:
    print("No objects detected in front.")

else:
    for detection in front_objects:
        print(
            f"{detection['object']} "
            f"(confidence: {detection['confidence']:.2f})"
        )


front_response = generate_front_objects_response(front_objects)

print("\nFront Object Response")
print("=" * 50)
print(front_response)

# Generate a natural-language scene summary
summary = summarize_scene(detections)

print("\nScene Summary")
print("=" * 50)
print(summary)
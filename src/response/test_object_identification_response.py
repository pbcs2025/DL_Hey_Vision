from ultralytics import YOLO

from src.reasoning.spatial_reasoning import process_yolo_detections
from src.reasoning.object_identification import identify_object
from src.response.object_identification_response import (
    generate_object_identification_response
)


MODEL_PATH = "backup/Experiment_02_Backup/best.pt"
IMAGE_PATH = "data/test/test_scene.jpg"


# Load YOLO model
model = YOLO(MODEL_PATH)


# Run object detection
results = model(
    IMAGE_PATH,
    conf=0.25,
    verbose=False
)


# Process detections
detections = process_yolo_detections(results)


# Identify the main object
identified_object = identify_object(detections)


# Generate assistant response
response = generate_object_identification_response(
    identified_object
)


print("\nAI Vision Assistant - Query 7")
print("=" * 60)

print("\nIdentified Object")
print("-" * 60)

if identified_object is None:
    print("No object identified.")
else:
    print(f"Object: {identified_object['object']}")
    print(f"Position: {identified_object['position']}")
    print(f"Confidence: {identified_object['confidence']:.2f}")

print("\nAssistant Response")
print("-" * 60)
print(response)

print("\nQuery 7 test completed.")
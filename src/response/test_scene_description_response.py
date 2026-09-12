from ultralytics import YOLO

from src.reasoning.spatial_reasoning import process_yolo_detections
from src.reasoning.scene_description import create_scene_description
from src.response.scene_description_response import (
    generate_scene_description_response
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


# Process YOLO detections
detections = process_yolo_detections(results)


# Create structured scene representation
scene = create_scene_description(detections)


# Generate natural-language description
response = generate_scene_description_response(scene)


print("\nAI Vision Assistant - Query 8")
print("=" * 60)

print("\nScene Representation")
print("-" * 60)

print("Object counts:", scene["object_counts"])
print("Total detections:", scene["total_objects"])

print("\nAssistant Response")
print("-" * 60)

print(response)

print("\nQuery 8 test completed.")
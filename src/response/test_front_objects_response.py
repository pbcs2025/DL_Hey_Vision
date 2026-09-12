from ultralytics import YOLO

from src.reasoning.spatial_reasoning import process_yolo_detections
from src.reasoning.directional_reasoning import get_front_objects
from src.response.front_objects_response import generate_front_objects_response


MODEL_PATH = "backup/Experiment_02_Backup/best.pt"
IMAGE_PATH = "data/test/test_scene.jpg"


# Load YOLO model
model = YOLO(MODEL_PATH)


# Detect objects
results = model(
    IMAGE_PATH,
    conf=0.25,
    verbose=False
)


# Process detections
detections = process_yolo_detections(results)


# Find objects in front
front_objects = get_front_objects(detections)


# Generate assistant response
response = generate_front_objects_response(front_objects)


print("\nAI Vision Assistant - Query 2")
print("=" * 60)

print("\nObjects in Front")
print("-" * 60)

for detection in front_objects:
    print(
        f"{detection['object']:15} | "
        f"{detection['confidence']:.2f}"
    )

print("\nAssistant Response")
print("-" * 60)
print(response)

print("\nQuery 2 test completed.")
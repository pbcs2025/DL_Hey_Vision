from ultralytics import YOLO

from src.reasoning.spatial_reasoning import process_yolo_detections
from src.reasoning.object_location import get_object_location
from src.response.object_location_response import generate_object_location_response


MODEL_PATH = "backup/Experiment_02_Backup/best.pt"
IMAGE_PATH = "data/test/door_test.png"


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


# Find door location
door_locations = get_object_location(
    detections,
    "door"
)


# Generate assistant response
response = generate_object_location_response(
    "door",
    door_locations
)


print("\nAI Vision Assistant - Query 3")
print("=" * 60)

print("\nDoor Location")
print("-" * 60)

if door_locations:
    for location in door_locations:
        print(f"Door: {location}")
else:
    print("Door not detected.")


print("\nAssistant Response")
print("-" * 60)
print(response)

print("\nQuery 3 test completed.")
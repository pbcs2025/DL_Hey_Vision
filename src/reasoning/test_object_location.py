from ultralytics import YOLO

from src.reasoning.spatial_reasoning import process_yolo_detections
from src.reasoning.object_location import get_object_location


MODEL_PATH = "backup/Experiment_02_Backup/best.pt"

# Use the DeepDoors2 test image later.
# For now, change this to the actual door image path you have locally.
IMAGE_PATH = "data/test/door_test.png"


# Load YOLO model
model = YOLO(MODEL_PATH)


# Run detection
results = model(
    IMAGE_PATH,
    conf=0.25,
    verbose=False
)


# Process YOLO detections
detections = process_yolo_detections(results)


# Find the location of the door
door_locations = get_object_location(
    detections,
    "door"
)


print("\nAI Vision Assistant - Query 3")
print("=" * 60)

print("\nAll Detected Objects")
print("-" * 60)

for detection in detections:
    print(
        f"{detection['object']:15} | "
        f"{detection['position']:6} | "
        f"{detection['confidence']:.2f}"
    )


print("\nDoor Location")
print("-" * 60)

if door_locations:
    for location in door_locations:
        print(f"Door: {location}")
else:
    print("Door not detected.")


print("\nQuery 3 object-location test completed.")
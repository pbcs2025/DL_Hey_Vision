from ultralytics import YOLO

from src.reasoning.spatial_reasoning import process_yolo_detections
from src.reasoning.directional_reasoning import get_front_objects


MODEL_PATH = "backup/Experiment_02_Backup/best.pt"
IMAGE_PATH = "data/test/test_scene.jpg"


# Load YOLO model
model = YOLO(MODEL_PATH)


# Run detection
results = model(
    IMAGE_PATH,
    conf=0.25,
    verbose=False
)


# Convert YOLO output into our detection format
detections = process_yolo_detections(results)


# Find objects in front
front_objects = get_front_objects(detections)


print("\nAI Vision Assistant - Query 2")
print("=" * 60)

print("\nAll Detected Objects")
print("-" * 60)

for detection in detections:
    print(
        f"{detection['object']:15} | "
        f"{detection['position']:6} | "
        f"{detection['confidence']:.2f}"
    )


print("\nObjects in Front")
print("-" * 60)

if not front_objects:
    print("No objects detected in front.")
else:
    for detection in front_objects:
        print(
            f"{detection['object']:15} | "
            f"{detection['position']:6} | "
            f"{detection['confidence']:.2f}"
        )


print("\nQuery 2 directional reasoning test completed.")
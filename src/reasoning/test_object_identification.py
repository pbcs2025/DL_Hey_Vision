from ultralytics import YOLO

from src.reasoning.spatial_reasoning import process_yolo_detections
from src.reasoning.object_identification import identify_object


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


# Identify the most confident object
identified_object = identify_object(detections)


print("\nAI Vision Assistant - Query 7")
print("=" * 60)

print("\nDetected Objects")
print("-" * 60)

for detection in detections:
    print(
        f"{detection['object']:15} | "
        f"{detection['position']:6} | "
        f"{detection['confidence']:.2f}"
    )


print("\nIdentified Object")
print("-" * 60)

if identified_object is None:
    print("No object identified.")
else:
    print(f"Object: {identified_object['object']}")
    print(f"Position: {identified_object['position']}")
    print(f"Confidence: {identified_object['confidence']:.2f}")


print("\nQuery 7 object-identification test completed.")
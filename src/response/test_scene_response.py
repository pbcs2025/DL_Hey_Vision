from ultralytics import YOLO

from src.reasoning.spatial_reasoning import process_yolo_detections
from src.response.scene_response import generate_scene_response


MODEL_PATH = "backup/Experiment_02_Backup/best.pt"
IMAGE_PATH = "data/test/test_scene.jpg"


model = YOLO(MODEL_PATH)

results = model(
    IMAGE_PATH,
    conf=0.25,
    verbose=False
)

detections = process_yolo_detections(results)

response = generate_scene_response(detections)


print("\nAI Vision Assistant - Query 1")
print("=" * 60)

print("\nDetected Objects")
print("-" * 60)

if not detections:
    print("No objects detected.")
else:
    object_counts = {}

    for detection in detections:
        object_name = detection["object"]

        if object_name not in object_counts:
            object_counts[object_name] = 0

        object_counts[object_name] += 1

    for object_name, count in object_counts.items():
        print(f"{count} {object_name}" if count > 1 else f"1 {object_name}")

print("\nAssistant Response")
print("-" * 60)
print(response)

print("\nQuery 1 test completed.")
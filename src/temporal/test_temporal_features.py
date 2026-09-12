from ..reasoning.spatial_reasoning import process_yolo_detections
from .feature_extraction import extract_temporal_features
from ultralytics import YOLO


model = YOLO(
    "backup/Experiment_02_Backup/best.pt"
)

image_path = "data/test/test_scene.jpg"

results = model(
    image_path,
    conf=0.25
)

detections = process_yolo_detections(results)

print("\nTemporal Features")
print("=" * 40)

for detection in detections:
    features = extract_temporal_features(
        detection,
        image_width=1200,
        image_height=1600
    )

    print(
        f"{detection['object']} → "
        f"{features}"
    )
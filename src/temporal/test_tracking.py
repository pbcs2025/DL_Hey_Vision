import cv2
from ultralytics import YOLO

MODEL_PATH = "backup/Experiment_02_Backup/best.pt"
VIDEO_PATH = "data/test/test_video.mp4"

model = YOLO(MODEL_PATH)

results = model.track(
    source=VIDEO_PATH,
    conf=0.25,
    persist=True,
    stream=True
)

print("\nObject Tracking Test")
print("=" * 50)

for frame_number, result in enumerate(results):

    if result.boxes is None or result.boxes.id is None:
        continue

    track_ids = result.boxes.id.int().cpu().tolist()
    class_ids = result.boxes.cls.int().cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()

    print(f"\nFrame {frame_number}")

    for track_id, class_id, confidence in zip(
        track_ids,
        class_ids,
        confidences
    ):
        class_name = result.names[class_id]

        print(
            f"Track ID: {track_id} | "
            f"Object: {class_name} | "
            f"Confidence: {confidence:.2f}"
        )
from ultralytics import YOLO

from .obstacle_reasoning import find_obstacles_ahead
from ..response.obstacle_response import generate_obstacle_response


MODEL_PATH = "backup/Experiment_02_Backup/best.pt"

IMAGE_PATH = "data/test/test_scene.jpg"


# Load YOLO model
model = YOLO(MODEL_PATH)


# Run object detection
results = model.predict(
    source=IMAGE_PATH,
    conf=0.25,
    verbose=False
)


# Get detections
result = results[0]

detections = []


if result.boxes is not None:

    image_height, image_width = result.orig_shape

    for box in result.boxes:

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        class_id = int(box.cls[0])

        confidence = float(box.conf[0])

        object_name = result.names[class_id]


        # Calculate bounding-box center
        x_center = (x1 + x2) / 2


        # Determine horizontal position
        normalized_x = x_center / image_width

        if normalized_x < 0.33:
            position = "left"

        elif normalized_x < 0.66:
            position = "center"

        else:
            position = "right"


        detections.append({
            "object": object_name,
            "confidence": confidence,
            "position": position
        })


# Find obstacles ahead
obstacles = find_obstacles_ahead(
    detections
)


# Generate response
response = generate_obstacle_response(
    obstacles
)


print("\nAI Vision Assistant - Query 6 Test")
print("=" * 60)


print("\nAll detections:")

for detection in detections:

    print(
        f"{detection['object']:15} | "
        f"{detection['position']:7} | "
        f"{detection['confidence']:.2f}"
    )


print("\nPotential obstacles ahead:")

if obstacles:

    for obstacle in obstacles:

        print(
            f"{obstacle['object']} | "
            f"{obstacle['position']} | "
            f"{obstacle['confidence']:.2f}"
        )

else:

    print("None")


print("\nAssistant:")
print(response)
import cv2
import numpy as np

from ultralytics import YOLO
from tensorflow.keras.models import load_model

from .track_history import TrackHistory
from .feature_extraction import extract_temporal_features

from ..response.movement_response import generate_movement_response
from ..response.alert_manager import AlertManager


MODEL_PATH = "backup/Experiment_02_Backup/best.pt"

LSTM_MODEL_PATH = "models/temporal/approaching_lstm.keras"

VIDEO_PATH = "data/test/test_video.mp4"

SEQUENCE_LENGTH = 30

PREDICTION_INTERVAL = 10

CLASS_NAMES = [
    "Stationary",
    "Approaching",
    "Moving Away",
    "Crossing"
]


# Load models
model = YOLO(MODEL_PATH)

lstm_model = load_model(LSTM_MODEL_PATH)


# Create tracking and alert managers
track_histories = TrackHistory(
    sequence_length=SEQUENCE_LENGTH
)

alert_manager = AlertManager()


# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

frame_number = 0

print("\nAI Vision Assistant - Query 5 Test")
print("=" * 60)


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1


    # YOLO tracking
    results = model.track(
        source=frame,
        conf=0.25,
        persist=True,
        verbose=False
    )

    result = results[0]


    # Skip frame if there are no tracked objects
    if result.boxes is None or result.boxes.id is None:
        continue


    image_height, image_width = result.orig_shape


    track_ids = result.boxes.id.int().cpu().tolist()

    class_ids = result.boxes.cls.int().cpu().tolist()

    confidences = result.boxes.conf.cpu().tolist()


    # Process each tracked object
    for index, (
        track_id,
        class_id,
        detection_confidence
    ) in enumerate(
        zip(
            track_ids,
            class_ids,
            confidences
        )
    ):

        class_name = result.names[class_id]


        # Currently monitoring people
        if class_name != "person":
            continue


        # Get bounding box
        box = result.boxes[index]

        x1, y1, x2, y2 = box.xyxy[0].tolist()


        # Create detection dictionary
        detection = {
            "x_center": (x1 + x2) / 2,
            "y_center": (y1 + y2) / 2,
            "width": x2 - x1,
            "height": y2 - y1
        }


        # Determine horizontal position
        normalized_x = detection["x_center"] / image_width

        if normalized_x < 0.33:
            position = "left"

        elif normalized_x < 0.66:
            position = "center"

        else:
            position = "right"


        # Extract normalized temporal features
        features = extract_temporal_features(
            detection,
            image_width,
            image_height
        )


        # Update history for this tracked person
        track_histories.update(
            track_id,
            features
        )


        # Get 30-frame sequence
        sequence = track_histories.get_sequence(
            track_id
        )


        # Wait until 30 frames are available
        if sequence is None:
            continue


        # Predict only every 10 frames
        if frame_number % PREDICTION_INTERVAL != 0:
            continue


        # Prepare LSTM input
        X = np.array(
            [sequence],
            dtype=np.float32
        )


        # LSTM prediction
        prediction = lstm_model.predict(
            X,
            verbose=0
        )


        predicted_class = np.argmax(
            prediction[0]
        )


        movement = CLASS_NAMES[
            predicted_class
        ]


        confidence_score = prediction[0][
            predicted_class
        ]


        # Generate spatial movement response
        response = generate_movement_response(
            movement,
            confidence_score,
            position
        )


        # State-based alert decision
        alert = alert_manager.should_alert(
            track_id,
            movement,
            confidence_score
        )


        # Update current movement state
        alert_manager.update_state(
            track_id,
            movement
        )


        # Print only when a new warning state is detected
        if alert and response is not None:

            print(
                f"\nFrame: {frame_number}"
            )

            print(
                f"Track ID: {track_id}"
            )

            print(
                f"Position: {position}"
            )

            print(
                f"Movement: {movement}"
            )

            print(
                f"Confidence: {confidence_score:.4f}"
            )

            print(
                f"Assistant: {response}"
            )


# Release video
cap.release()

print("\nQuery 5 test completed.")
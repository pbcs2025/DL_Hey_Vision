import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

from src.temporal.feature_extraction import (
    extract_temporal_features,
    build_sequence
)

from src.risk.test_risk_dqn import predict_risk
from src.dqn.dqn_bridge import get_dqn_action


# --------------------------------------------------
# Paths
# --------------------------------------------------

YOLO_MODEL_PATH = "backup/Experiment_02_Backup/best.pt"
LSTM_MODEL_PATH = "models/temporal/approaching_lstm.keras"
VIDEO_PATH = "data/test/test_video.mp4"


# --------------------------------------------------
# Load models
# --------------------------------------------------

yolo_model = YOLO(YOLO_MODEL_PATH)

lstm_model = load_model(
    LSTM_MODEL_PATH,
    compile=False
)


# --------------------------------------------------
# Movement labels
# --------------------------------------------------

MOVEMENT_LABELS = [
    "Stationary",
    "Approaching",
    "Moving Away",
    "Crossing"
]


# --------------------------------------------------
# Track feature histories
# --------------------------------------------------

track_histories = {}


# --------------------------------------------------
# Open video
# --------------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(
        f"Could not open: {VIDEO_PATH}"
    )


frame_number = 0


print()
print("AI Vision Assistant")
print("YOLO → Tracking → LSTM → Risk ANN → DQN")
print("=" * 65)


# --------------------------------------------------
# Process video
# --------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # --------------------------------------------------
    # YOLO tracking
    # --------------------------------------------------

    results = yolo_model.track(
        frame,
        persist=True,
        conf=0.25,
        verbose=False
    )

    for result in results:

        if result.boxes is None:
            continue

        if result.boxes.id is None:
            continue

        track_ids = (
            result.boxes.id
            .int()
            .cpu()
            .tolist()
        )

        for box, track_id in zip(
            result.boxes,
            track_ids
        ):

            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            object_name = result.names[class_id]

            # --------------------------------------------------
            # For this test, track people
            # --------------------------------------------------

            if object_name != "person":
                continue

            # --------------------------------------------------
            # Create detection dictionary
            # --------------------------------------------------

            detection = {
                "x_center": (x1 + x2) / 2,
                "y_center": (y1 + y2) / 2,
                "width": x2 - x1,
                "height": y2 - y1
            }

            # --------------------------------------------------
            # Extract temporal features
            # --------------------------------------------------

            features = extract_temporal_features(
                detection,
                frame.shape[1],
                frame.shape[0]
            )

            # --------------------------------------------------
            # Add features to this track's history
            # --------------------------------------------------

            if track_id not in track_histories:
                track_histories[track_id] = []

            track_histories[track_id].append(
                features
            )

            # --------------------------------------------------
            # Keep only recent history
            # --------------------------------------------------

            if len(track_histories[track_id]) > 30:
                track_histories[track_id] = (
                    track_histories[track_id][-30:]
                )

            # --------------------------------------------------
            # Build 30-frame sequence
            # --------------------------------------------------

            sequence = build_sequence(
                track_histories[track_id],
                sequence_length=30
            )

            if sequence is None:
                continue

            sequence = np.array(
                sequence,
                dtype=np.float32
            )

            sequence = np.expand_dims(
                sequence,
                axis=0
            )

            # --------------------------------------------------
            # LSTM prediction
            # --------------------------------------------------

            prediction = lstm_model.predict(
                sequence,
                verbose=0
            )[0]

            movement_id = int(
                np.argmax(prediction)
            )

            movement = MOVEMENT_LABELS[
                movement_id
            ]

            movement_confidence = float(
                prediction[movement_id]
            )

            # --------------------------------------------------
            # Spatial position
            # --------------------------------------------------

            x_center = (
                (x1 + x2) / 2
            )

            normalized_x = (
                x_center / frame.shape[1]
            )

            if normalized_x < 0.33:
                position = "left"

            elif normalized_x < 0.66:
                position = "center"

            else:
                position = "right"

            ahead = position == "center"

            # --------------------------------------------------
            # Risk ANN
            # --------------------------------------------------

            risk = predict_risk(
                object_name=object_name,
                position=position,
                width=x2 - x1,
                height=y2 - y1,
                image_width=frame.shape[1],
                image_height=frame.shape[0],
                confidence=confidence,
                movement=movement.lower(),
                ahead=ahead
            )

            # --------------------------------------------------
            # DQN
            # --------------------------------------------------

            decision = get_dqn_action(
                risk=risk["risk_score"],
                confidence=confidence,
                movement=movement.lower(),
                ahead=int(ahead),
                intent="DETECT_APPROACHING",
                recent_alert=0
            )

            # --------------------------------------------------
            # Print result
            # --------------------------------------------------

            print(
                f"Frame {frame_number:03d} | "
                f"Track {track_id} | "
                f"{position:6s} | "
                f"{movement:11s} "
                f"({movement_confidence:.3f}) | "
                f"Risk: {risk['risk_name']:8s} | "
                f"DQN: {decision['action']}"
            )


# --------------------------------------------------
# Finish
# --------------------------------------------------

cap.release()

print()
print("Temporal integration test completed.")
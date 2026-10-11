import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO

from src.temporal.feature_extraction import (
    extract_temporal_features,
    build_sequence
)


# --------------------------------------------------
# Model paths
# --------------------------------------------------

YOLO_MODEL_PATH = (
    "backup/Experiment_02_Backup/best.pt"
)

LSTM_MODEL_PATH = (
    "models/temporal/approaching_lstm.keras"
)

VIDEO_PATH = (
    "data/test/test_video.mp4"
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
# Load models
# --------------------------------------------------

yolo_model = YOLO(
    YOLO_MODEL_PATH
)

lstm_model = load_model(
    LSTM_MODEL_PATH,
    compile=False
)


# --------------------------------------------------
# Track histories
# --------------------------------------------------

track_histories = {}


# --------------------------------------------------
# Predict movement
# --------------------------------------------------

def predict_movement(sequence):

    sequence_array = np.array(
        sequence,
        dtype=np.float32
    )

    sequence_array = np.expand_dims(
        sequence_array,
        axis=0
    )

    prediction = lstm_model.predict(
        sequence_array,
        verbose=0
    )[0]

    movement_id = int(
        np.argmax(prediction)
    )

    confidence = float(
        prediction[movement_id]
    )

    return (
        MOVEMENT_LABELS[movement_id],
        confidence
    )


# --------------------------------------------------
# Process video
# --------------------------------------------------

def process_video():

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():

        print(
            "Error: Could not open video."
        )

        return


    frame_count = 0

    print("\nAI Vision Assistant")
    print("=" * 60)

    print(
        "YOLO Tracking → LSTM Movement Prediction"
    )

    print("=" * 60)


    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_count += 1


        # ------------------------------------------
        # YOLO tracking
        # ------------------------------------------

        results = yolo_model.track(
            frame,
            persist=True,
            conf=0.25,
            verbose=False
        )


        if not results:
            continue


        result = results[0]


        if (
            result.boxes is None
            or result.boxes.id is None
        ):
            continue


        track_ids = (
            result.boxes.id
            .int()
            .cpu()
            .tolist()
        )


        # ------------------------------------------
        # Process each tracked object
        # ------------------------------------------

        for index, track_id in enumerate(
            track_ids
        ):

            box = result.boxes[index]


            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .tolist()
            )


            confidence = float(
                box.conf[0]
                .cpu()
            )


            class_id = int(
                box.cls[0]
                .cpu()
            )


            object_name = (
                result.names[class_id]
            )


            # --------------------------------------
            # Build detection dictionary
            # --------------------------------------

            image_height, image_width = (
                frame.shape[:2]
            )


            x_center = (
                x1 + x2
            ) / 2.0

            y_center = (
                y1 + y2
            ) / 2.0

            width = (
                x2 - x1
            )

            height = (
                y2 - y1
            )


            detection = {

                "object":
                    object_name,

                "confidence":
                    confidence,

                "x_center":
                    x_center,

                "y_center":
                    y_center,

                "width":
                    width,

                "height":
                    height
            }


            # --------------------------------------
            # Extract temporal features
            # --------------------------------------

            features = extract_temporal_features(

                detection,

                image_width,

                image_height
            )


            # --------------------------------------
            # Store track history
            # --------------------------------------

            if track_id not in track_histories:

                track_histories[track_id] = []


            track_histories[
                track_id
            ].append(features)


            # --------------------------------------
            # Keep history manageable
            # --------------------------------------

            if len(
                track_histories[track_id]
            ) > 30:

                track_histories[
                    track_id
                ] = track_histories[
                    track_id
                ][-30:]


            # --------------------------------------
            # Build LSTM sequence
            # --------------------------------------

            sequence = build_sequence(

                track_histories[track_id],

                sequence_length=30
            )


            if sequence is None:

                continue


            # --------------------------------------
            # Predict movement
            # --------------------------------------

            movement, movement_confidence = (
                predict_movement(sequence)
            )


            # --------------------------------------
            # Print prediction
            # --------------------------------------

            if (
                frame_count % 15 == 0
            ):

                print(

                    f"Frame {frame_count} | "

                    f"Track {track_id} | "

                    f"{object_name} | "

                    f"{movement} | "

                    f"{movement_confidence:.3f}"
                )


    cap.release()


    print("\n" + "=" * 60)

    print(
        f"Total frames processed: "
        f"{frame_count}"
    )

    print(
        "Temporal pipeline test completed."
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    process_video()
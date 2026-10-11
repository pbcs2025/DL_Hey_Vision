import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from ultralytics import YOLO

from src.temporal.feature_extraction import (
    extract_temporal_features,
    build_sequence
)

from src.temporal.alert_manager import (
    AlertManager
)

from src.risk.risk_features import (
    create_risk_features
)

from src.dqn.dqn_bridge import (
    get_dqn_action
)


from src.pipeline.model_lock import YOLO_LOCK, KERAS_LOCK

# --------------------------------------------------
# Model paths
# --------------------------------------------------

YOLO_MODEL_PATH = (
    "backup/Experiment_02_Backup/best.pt"
)

LSTM_MODEL_PATH = (
    "models/temporal/approaching_lstm.keras"
)

RISK_MODEL_PATH = (
    "models/risk/risk_ann.keras"
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
# Risk labels
# --------------------------------------------------

RISK_LABELS = [
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL"
]


# --------------------------------------------------
# Load models
# --------------------------------------------------

print("Loading models...")

with YOLO_LOCK:
    yolo_model = YOLO(
        YOLO_MODEL_PATH
    )

with KERAS_LOCK:
    lstm_model = load_model(
        LSTM_MODEL_PATH,
        compile=False
    )

    risk_model = load_model(
        RISK_MODEL_PATH,
        compile=False
    )

print("Models loaded successfully.")


# --------------------------------------------------
# Predict movement using LSTM (Direct callable, no .predict() overhead)
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

    with KERAS_LOCK:
        tensor_in = tf.convert_to_tensor(sequence_array)
        prediction = lstm_model(
            tensor_in,
            training=False
        ).numpy()[0]

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
# Predict risk using Risk ANN (Direct callable)
# --------------------------------------------------

def predict_risk(
    detection,
    movement,
    ahead
):

    image_width = detection[
        "image_width"
    ]

    image_height = detection[
        "image_height"
    ]

    features = create_risk_features(

        object_name=detection["object"],

        position=detection["position"],

        width=detection["width"],

        height=detection["height"],

        image_width=image_width,

        image_height=image_height,

        confidence=detection["confidence"],

        movement=movement,

        ahead=ahead
    )

    with KERAS_LOCK:
        tensor_in = tf.convert_to_tensor(
            [features],
            dtype=tf.float32
        )
        prediction = risk_model(
            tensor_in,
            training=False
        ).numpy()[0]

    risk_id = int(
        np.argmax(prediction)
    )

    risk_confidence = float(
        prediction[risk_id]
    )

    return (
        RISK_LABELS[risk_id],
        risk_confidence,
        risk_id
    )


# --------------------------------------------------
# DQN decision
# --------------------------------------------------

def make_dqn_decision(

    detection,

    risk_id,

    movement,

    recent_alert=0

):

    risk_score = (
        risk_id / 3.0
    )

    ahead = (

        1

        if detection["position"]
        == "center"

        else 0
    )

    decision = get_dqn_action(

        risk=risk_score,

        confidence=detection[
            "confidence"
        ],

        movement=movement,

        ahead=ahead,

        intent="DETECT_APPROACHING",

        recent_alert=recent_alert
    )

    return decision


# --------------------------------------------------
# Process video
# --------------------------------------------------

def process_video(
    video_path=VIDEO_PATH,
    progress_callback=None,
    stride: int = 1,
    imgsz: int = 640,
    cancel_check=None
):

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        print(
            "Error: Could not open video."
        )

        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    # Reset histories for each new video
    track_histories = {}


    # Reset alert manager for each new video
    alert_manager = AlertManager()


    frame_count = 0


    # Store important movement events
    movement_events = []


    print("\n")

    print("AI Vision Assistant")
    print("=" * 70)

    print(
        "YOLO → Tracking → LSTM → "
        "Risk ANN → AlertManager → DQN"
    )

    print("=" * 70)


    while True:

        success, frame = cap.read()

        if not success:
            break


        frame_count += 1

        if cancel_check and callable(cancel_check) and cancel_check():
            print(f"[VideoAssistantPipeline] Cancelled by user at frame {frame_count}.")
            break

        if stride > 1 and (frame_count - 1) % stride != 0:
            if progress_callback and callable(progress_callback):
                try:
                    progress_callback(frame_count, total_frames)
                except Exception:
                    pass
            continue

        if progress_callback and callable(progress_callback):
            try:
                progress_callback(frame_count, total_frames)
            except Exception:
                pass


        # ------------------------------------------
        # YOLO tracking
        # ------------------------------------------

        with YOLO_LOCK:
            results = yolo_model.track(

                frame,

                persist=True,

                conf=0.25,

                verbose=False,

                imgsz=imgsz
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


        image_height, image_width = (
            frame.shape[:2]
        )


        # ------------------------------------------
        # Process tracked objects
        # ------------------------------------------

        for index, track_id in enumerate(
            track_ids
        ):

            box = result.boxes[index]


            # --------------------------------------
            # Bounding box
            # --------------------------------------

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
            # Bounding-box features
            # --------------------------------------

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


            # --------------------------------------
            # Horizontal position
            # --------------------------------------

            normalized_x = (

                x_center
                / image_width
            )


            if normalized_x < 0.33:

                position = "left"

            elif normalized_x < 0.66:

                position = "center"

            else:

                position = "right"


            # --------------------------------------
            # Detection dictionary
            # --------------------------------------

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
                    height,

                "position":
                    position,

                "image_width":
                    image_width,

                "image_height":
                    image_height
            }


            # --------------------------------------
            # Temporal features
            # --------------------------------------

            features = (
                extract_temporal_features(

                    detection,

                    image_width,

                    image_height
                )
            )


            # --------------------------------------
            # Track history
            # --------------------------------------

            if track_id not in track_histories:

                track_histories[
                    track_id
                ] = []


            track_histories[
                track_id
            ].append(
                features
            )


            # Keep only the latest 30 frames
            if len(
                track_histories[track_id]
            ) > 30:

                track_histories[
                    track_id
                ] = track_histories[
                    track_id
                ][-30:]


            # --------------------------------------
            # Build sequence
            # --------------------------------------

            sequence = build_sequence(

                track_histories[
                    track_id
                ],

                sequence_length=30
            )


            if sequence is None:

                continue


            # --------------------------------------
            # LSTM movement prediction
            # --------------------------------------

            (
                movement,
                movement_confidence

            ) = predict_movement(
                sequence
            )


            # --------------------------------------
            # Determine ahead status
            # --------------------------------------

            ahead = (

                1

                if position == "center"

                else 0
            )


            # --------------------------------------
            # Risk ANN
            # --------------------------------------

            (
                risk_label,
                risk_confidence,
                risk_id

            ) = predict_risk(

                detection=detection,

                movement=movement,

                ahead=ahead
            )


            # --------------------------------------
            # AlertManager
            # --------------------------------------

            new_alert = (
                alert_manager.should_alert(

                    track_id=track_id,

                    movement=movement,

                    confidence=movement_confidence
                )
            )


            # --------------------------------------
            # Recent alert state
            # --------------------------------------

            recent_alert = (

                1

                if new_alert

                else 0
            )


            # --------------------------------------
            # DQN
            # --------------------------------------

            decision = make_dqn_decision(

                detection=detection,

                risk_id=risk_id,

                movement=movement,

                recent_alert=recent_alert
            )


            # --------------------------------------
            # Final action after AlertManager
            # --------------------------------------

            dqn_action = decision["action"]


            if dqn_action == "WARN":

                if new_alert:

                    final_action = "WARN"

                else:

                    final_action = "WAIT"

            else:

                final_action = dqn_action


            # --------------------------------------
            # Update AlertManager
            # --------------------------------------

            alert_manager.update_state(

                track_id,

                movement
            )


            # --------------------------------------
            # Store and print important events
            # --------------------------------------

            if (

                new_alert

                or final_action == "WARN"

            ):

                event = {

                    "frame":
                        frame_count,

                    "track_id":
                        track_id,

                    "object":
                        object_name,

                    "position":
                        position,

                    "movement":
                        movement,

                    "movement_confidence":
                        movement_confidence,

                    "risk":
                        risk_label,

                    "risk_confidence":
                        risk_confidence,

                    "dqn_action":
                        dqn_action,

                    "final_action":
                        final_action
                }


                movement_events.append(
                    event
                )


                print(
                    "\n"
                    + "-" * 70
                )


                print(
                    f"Frame: {frame_count}"
                )


                print(
                    f"Track ID: {track_id}"
                )


                print(
                    f"Object: {object_name}"
                )


                print(
                    f"Position: {position}"
                )


                print(
                    f"Movement: {movement} "
                    f"({movement_confidence:.3f})"
                )


                print(
                    f"Risk: {risk_label} "
                    f"({risk_confidence:.3f})"
                )


                print(
                    f"New movement alert: "
                    f"{new_alert}"
                )


                print(
                    f"DQN Decision: "
                    f"{dqn_action}"
                )


                print(
                    f"Final Action: "
                    f"{final_action}"
                )


    # ------------------------------------------------
    # Release video
    # ------------------------------------------------

    cap.release()


    print(
        "\n"
        + "=" * 70
    )


    print(
        f"Total frames processed: "
        f"{frame_count}"
    )


    print(
        "Video pipeline with AlertManager "
        "completed."
    )


    # ------------------------------------------------
    # Return events so another pipeline can use them
    # ------------------------------------------------

    return movement_events


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    process_video()
from pathlib import Path
from collections import defaultdict, deque

import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

from src.vision.ocr import read_text
from src.nlp.intent_labels import INTENTS

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

YOLO_MODEL_PATH = (
    PROJECT_ROOT
    / "backup"
    / "Experiment_02_Backup"
    / "best.pt"
)

NLP_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "nlp"
    / "query_embedding_model_expanded.keras"
)

LSTM_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "temporal"
    / "approaching_lstm.keras"
)

RISK_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "risk"
    / "risk_ann.keras"
)

DQN_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "dqn"
    / "assistive_dqn.keras"
)


# ============================================================
# MODEL LABELS
# ============================================================

MOVEMENT_LABELS = [
    "stationary",
    "approaching",
    "moving_away",
    "crossing"
]

MOVEMENT_ENCODING = {
    "stationary": 0.0,
    "moving_away": 0.25,
    "crossing": 0.75,
    "approaching": 1.0
}

RISK_LABELS = [
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL"
]

RISK_ENCODING = {
    "LOW": 0.0,
    "MEDIUM": 0.33,
    "HIGH": 0.66,
    "CRITICAL": 1.0
}

ACTION_LABELS = {
    0: "WAIT",
    1: "SPEAK",
    2: "WARN"
}

INTENT_LABELS = INTENTS


# ============================================================
# QUERY PRIORITY
# ============================================================

QUERY_PRIORITY = {
    "SCENE_SUMMARY": 0.4,
    "FRONT_OBJECTS": 0.6,
    "LOCATE_OBJECT": 0.6,
    "READ_TEXT": 0.5,
    "DETECT_APPROACHING": 1.0,
    "OBSTACLE_CHECK": 1.0,
    "OBJECT_IDENTIFICATION": 0.5,
    "SCENE_DESCRIPTION": 0.5,
    "REPEAT_RESPONSE": 0.4,
    "MORE_DETAIL": 0.4
}


# ============================================================
# ASSISTIVE VISION PIPELINE
# ============================================================

class AssistiveVisionPipeline:

    def __init__(self):

        print("\nLoading AI Vision Assistant models...")

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------
        self.yolo = YOLO(str(YOLO_MODEL_PATH))

        # ----------------------------------------------------
        # NLP
        # ----------------------------------------------------
        self.nlp_model = tf.keras.models.load_model(
            NLP_MODEL_PATH
        )

        # ----------------------------------------------------
        # LSTM
        # ----------------------------------------------------
        self.lstm_model = tf.keras.models.load_model(
            LSTM_MODEL_PATH
        )

        # ----------------------------------------------------
        # Risk ANN
        # ----------------------------------------------------
        self.risk_model = tf.keras.models.load_model(
            RISK_MODEL_PATH
        )

        # ----------------------------------------------------
        # DQN
        # ----------------------------------------------------
        self.dqn_model = tf.keras.models.load_model(
            DQN_MODEL_PATH,
            compile=False
        )

        # ----------------------------------------------------
        # Temporal histories
        # ----------------------------------------------------
        self.track_histories = defaultdict(
            lambda: deque(maxlen=30)
        )

        # ----------------------------------------------------
        # Alert state
        # ----------------------------------------------------
        self.recent_alerts = {}

        print("All models loaded successfully.\n")


    # ========================================================
    # NLP QUERY UNDERSTANDING
    # ========================================================

    def understand_query(self, query):

        prediction = self.nlp_model.predict(
            np.array([query]),
            verbose=0
        )

        intent_id = int(np.argmax(prediction[0]))

        intent = INTENT_LABELS[intent_id]

        confidence = float(
            prediction[0][intent_id]
        )

        return {
            "intent": intent,
            "confidence": confidence
        }


    # ========================================================
    # YOLO DETECTION
    # ========================================================

    def detect_objects(self, frame):

        results = self.yolo(
            frame,
            conf=0.25,
            verbose=False
        )

        detections = []

        height, width = frame.shape[:2]

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                x1, y1, x2, y2 = (
                    box.xyxy[0].tolist()
                )

                class_id = int(box.cls[0])

                confidence = float(
                    box.conf[0]
                )

                class_name = result.names[class_id]

                box_width = x2 - x1
                box_height = y2 - y1

                x_center = (
                    x1 + x2
                ) / 2

                y_center = (
                    y1 + y2
                ) / 2

                normalized_x = (
                    x_center / width
                )

                if normalized_x < 0.33:
                    position = "left"

                elif normalized_x < 0.66:
                    position = "center"

                else:
                    position = "right"

                normalized_size = (
                    box_width * box_height
                ) / (
                    width * height
                )

                detections.append({
                    "object": class_name,
                    "class_id": class_id,
                    "confidence": confidence,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "x_center": x_center,
                    "y_center": y_center,
                    "position": position,
                    "width": box_width,
                    "height": box_height,
                    "normalized_size": normalized_size
                })

        return detections


    # ========================================================
    # SPATIAL REASONING
    # ========================================================

    def get_front_objects(self, detections):

        return [
            d for d in detections
            if d["position"] == "center"
        ]


    def find_object(self, detections, object_name):

        matches = [
            d for d in detections
            if d["object"].lower()
            == object_name.lower()
        ]

        if not matches:
            return None

        return max(
            matches,
            key=lambda x: x["confidence"]
        )


    # ========================================================
    # TEMPORAL FEATURE EXTRACTION
    # ========================================================

    def build_temporal_feature(self, detection, frame):

        height, width = frame.shape[:2]

        return [
            detection["x_center"] / width,
            detection["y_center"] / height,
            detection["width"] / width,
            detection["height"] / height
        ]


    # ========================================================
    # LSTM MOVEMENT PREDICTION
    # ========================================================

    def predict_movement(
        self,
        track_id,
        detection,
        frame
    ):

        feature = self.build_temporal_feature(
            detection,
            frame
        )

        self.track_histories[track_id].append(
            feature
        )

        history = self.track_histories[track_id]

        if len(history) < 30:

            return {
                "movement": "stationary",
                "confidence": 0.0,
                "ready": False
            }

        sequence = np.array(
            history,
            dtype=np.float32
        )

        sequence = np.expand_dims(
            sequence,
            axis=0
        )

        prediction = self.lstm_model.predict(
            sequence,
            verbose=0
        )

        movement_id = int(
            np.argmax(prediction[0])
        )

        movement = MOVEMENT_LABELS[
            movement_id
        ]

        confidence = float(
            prediction[0][movement_id]
        )

        return {
            "movement": movement,
            "confidence": confidence,
            "ready": True
        }


    # ========================================================
    # RISK ANN
    # ========================================================

    def predict_risk(
        self,
        detection,
        movement,
        ahead
    ):

        # Object type normalized
        object_feature = (
            detection["class_id"] / 12.0
        )

        # Spatial position normalized
        position_encoding = {
            "left": 0.0,
            "center": 0.5,
            "right": 1.0
        }

        position_feature = position_encoding[
            detection["position"]
        ]

        size_feature = min(
            detection["normalized_size"],
            1.0
        )

        confidence_feature = (
            detection["confidence"]
        )

        movement_feature = (
            MOVEMENT_ENCODING[movement]
        )

        ahead_feature = (
            1.0 if ahead else 0.0
        )

        features = np.array(
            [[
                object_feature,
                position_feature,
                size_feature,
                confidence_feature,
                movement_feature,
                ahead_feature
            ]],
            dtype=np.float32
        )

        prediction = self.risk_model.predict(
            features,
            verbose=0
        )

        risk_id = int(
            np.argmax(prediction[0])
        )

        risk = RISK_LABELS[risk_id]

        confidence = float(
            prediction[0][risk_id]
        )

        return {
            "risk": risk,
            "confidence": confidence
        }


    # ========================================================
    # DQN DECISION
    # ========================================================

    def make_decision(
        self,
        risk,
        detection_confidence,
        movement,
        ahead,
        query_priority,
        recent_alert
    ):

        state = np.array(
            [[
                RISK_ENCODING[risk],
                detection_confidence,
                MOVEMENT_ENCODING[movement],
                1.0 if ahead else 0.0,
                query_priority,
                1.0 if recent_alert else 0.0
            ]],
            dtype=np.float32
        )

        q_values = self.dqn_model.predict(
            state,
            verbose=0
        )[0]

        action_id = int(
            np.argmax(q_values)
        )

        return {
            "action": ACTION_LABELS[action_id],
            "action_id": action_id,
            "q_values": q_values.tolist()
        }


    # ========================================================
    # RESPONSE GENERATION
    # ========================================================

    def generate_response(
        self,
        intent,
        detections,
        target=None
    ):

        # ----------------------------------------------------
        # Scene summary
        # ----------------------------------------------------

        if intent == "SCENE_SUMMARY":

            if not detections:
                return "I cannot detect any objects around you."

            counts = {}

            for d in detections:
                obj = d["object"]
                counts[obj] = counts.get(obj, 0) + 1

            parts = []

            for obj, count in counts.items():

                if count == 1:
                    parts.append(
                        f"a {obj}"
                    )
                else:
                    parts.append(
                        f"{count} {obj}s"
                    )

            return (
                "I can see "
                + ", ".join(parts[:-1])
                + (" and " if len(parts) > 1 else "")
                + parts[-1]
                + " around you."
            )


        # ----------------------------------------------------
        # Front objects
        # ----------------------------------------------------

        if intent == "FRONT_OBJECTS":

            front = self.get_front_objects(
                detections
            )

            if not front:
                return "I do not detect anything directly in front of you."

            objects = {}

            for d in front:
                obj = d["object"]
                objects[obj] = (
                    objects.get(obj, 0) + 1
                )

            parts = []

            for obj, count in objects.items():

                if count == 1:
                    parts.append(
                        f"a {obj}"
                    )
                else:
                    parts.append(
                        f"{count} {obj}s"
                    )

            return (
                "I can see "
                + ", ".join(parts)
                + " in front of you."
            )


        # ----------------------------------------------------
        # Locate object
        # ----------------------------------------------------

        if intent == "LOCATE_OBJECT":

            if target is None:
                return "I could not identify the object you asked for."

            return (
                f"The {target['object']} "
                f"is on your {target['position']}."
            )


        # ----------------------------------------------------
        # Object identification
        # ----------------------------------------------------

        if intent == "OBJECT_IDENTIFICATION":

            if not detections:
                return "I could not identify the object."

            best = max(
                detections,
                key=lambda x: x["confidence"]
            )

            return (
                f"This appears to be a "
                f"{best['object']}."
            )


        # ----------------------------------------------------
        # Obstacle
        # ----------------------------------------------------

        if intent == "OBSTACLE_CHECK":

            front = self.get_front_objects(
                detections
            )

            if not front:
                return "I do not detect an obstacle directly ahead."

            objects = [
                d["object"]
                for d in front
            ]

            return (
                "Yes, I detect "
                + ", ".join(objects)
                + " directly ahead."
            )


        # ----------------------------------------------------
        # Scene description
        # ----------------------------------------------------

        if intent == "SCENE_DESCRIPTION":

            if not detections:
                return "The scene is unclear."

            objects = list(
                dict.fromkeys(
                    d["object"]
                    for d in detections
                )
            )

            return (
                "The scene contains "
                + ", ".join(objects)
                + "."
            )


        return "I am processing the scene."


    # ========================================================
    # SINGLE IMAGE QUERY
    # ========================================================

    def process_image(
        self,
        frame,
        query
    ):

        # ----------------------------------------------------
        # Step 1: Understand query
        # ----------------------------------------------------

        query_info = self.understand_query(
            query
        )

        intent = query_info["intent"]

        # ----------------------------------------------------
        # Step 2: Vision
        # ----------------------------------------------------

        detections = self.detect_objects(
            frame
        )

        # ----------------------------------------------------
        # Step 3: Read text
        # ----------------------------------------------------

        if intent == "READ_TEXT":

            text = read_text(frame)

            if text:
                return {
                    "intent": intent,
                    "response":
                        f"The sign says {text}."
                }

            return {
                "intent": intent,
                "response":
                    "I could not read any text."
            }

        # ----------------------------------------------------
        # Step 4: Find target
        # ----------------------------------------------------

        target = None

        if intent == "LOCATE_OBJECT":

            query_lower = query.lower()

            for d in detections:

                if d["object"].lower() in query_lower:

                    target = d

                    break

        # ----------------------------------------------------
        # Step 5: Response
        # ----------------------------------------------------

        response = self.generate_response(
            intent,
            detections,
            target
        )

        return {
            "intent": intent,
            "intent_confidence":
                query_info["confidence"],
            "detections": detections,
            "response": response
        }


    # ========================================================
    # VIDEO SAFETY PIPELINE
    # ========================================================

    def process_video(
        self,
        video_path,
        query="Is anything approaching me?"
    ):

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():

            raise RuntimeError(
                f"Could not open video: {video_path}"
            )

        query_info = self.understand_query(
            query
        )

        intent = query_info["intent"]

        query_priority = QUERY_PRIORITY.get(
            intent,
            0.5
        )

        frame_count = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_count += 1

            # ------------------------------------------------
            # YOLO tracking
            # ------------------------------------------------

            results = self.yolo.track(
                frame,
                persist=True,
                conf=0.25,
                verbose=False
            )

            if not results:
                continue

            result = results[0]

            if result.boxes is None:
                continue

            for box in result.boxes:

                if box.id is None:
                    continue

                track_id = int(
                    box.id[0]
                )

                x1, y1, x2, y2 = (
                    box.xyxy[0].tolist()
                )

                class_id = int(box.cls[0])

                confidence = float(
                    box.conf[0]
                )

                class_name = (
                    result.names[class_id]
                )

                height, width = (
                    frame.shape[:2]
                )

                x_center = (
                    x1 + x2
                ) / 2

                y_center = (
                    y1 + y2
                ) / 2

                normalized_x = (
                    x_center / width
                )

                if normalized_x < 0.33:
                    position = "left"

                elif normalized_x < 0.66:
                    position = "center"

                else:
                    position = "right"

                box_width = x2 - x1
                box_height = y2 - y1

                normalized_size = (
                    box_width * box_height
                ) / (
                    width * height
                )

                detection = {
                    "object": class_name,
                    "class_id": class_id,
                    "confidence": confidence,
                    "x_center": x_center,
                    "y_center": y_center,
                    "position": position,
                    "width": box_width,
                    "height": box_height,
                    "normalized_size":
                        normalized_size
                }

                # --------------------------------------------
                # Temporal model
                # --------------------------------------------

                movement_info = (
                    self.predict_movement(
                        track_id,
                        detection,
                        frame
                    )
                )

                movement = movement_info[
                    "movement"
                ]

                # --------------------------------------------
                # Spatial reasoning
                # --------------------------------------------

                ahead = (
                    position == "center"
                )

                # --------------------------------------------
                # Risk ANN
                # --------------------------------------------

                risk_info = self.predict_risk(
                    detection,
                    movement,
                    ahead
                )

                risk = risk_info["risk"]

                # --------------------------------------------
                # Previous alert?
                # --------------------------------------------

                recent_alert = self.recent_alerts.get(
                    track_id,
                    False
                )

                # --------------------------------------------
                # DQN
                # --------------------------------------------

                decision = self.make_decision(
                    risk=risk,
                    detection_confidence=
                        confidence,
                    movement=movement,
                    ahead=ahead,
                    query_priority=
                        query_priority,
                    recent_alert=
                        recent_alert
                )

                action = decision["action"]

                # --------------------------------------------
                # Update alert state
                # --------------------------------------------

                self.recent_alerts[track_id] = (
                    action == "WARN"
                )

                # --------------------------------------------
                # Print important decisions
                # --------------------------------------------

                if action in ["WARN", "SPEAK"]:

                    print(
                        f"Frame {frame_count} | "
                        f"Track {track_id} | "
                        f"{class_name} | "
                        f"{position} | "
                        f"{movement} | "
                        f"{risk} | "
                        f"{action}"
                    )

        cap.release()

        print(
            f"\nVideo processing completed. "
            f"Frames processed: {frame_count}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    pipeline = AssistiveVisionPipeline()

    print("AI Vision Assistant ready.")

    # Example image test:
    #
    # image = cv2.imread(
    #     "data/test/door_test.png"
    # )
    #
    # result = pipeline.process_image(
    #     image,
    #     "Where is the door?"
    # )
    #
    # print(result["response"])

    # Example video test:
    #
    # pipeline.process_video(
    #     "data/test/test_video.mp4",
    #     "Is anything approaching me?"
    # )
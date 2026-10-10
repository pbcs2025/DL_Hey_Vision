import os
import time
import tensorflow as tf

from tensorflow.keras.models import load_model
from ultralytics import YOLO

from src.pipeline.model_lock import YOLO_LOCK, KERAS_LOCK, WHISPER_LOCK, MODEL_LOCK
from src.risk.risk_features import create_risk_features
from src.dqn.dqn_bridge import get_dqn_action

from src.reasoning.spatial_reasoning import process_yolo_detections
from src.reasoning.directional_reasoning import get_front_objects
from src.reasoning.object_location import get_object_location
from src.reasoning.object_identification import identify_object

from src.vision.ocr import read_text

from src.response.scene_summary import summarize_scene
from src.response.scene_description_response import (
    generate_scene_description_response
)

from src.response.conversation_state import (
    save_state,
    get_last_response,
    get_last_detections
)

from src.response.more_detail_response import (
    generate_more_detail_response
)

from src.pipeline.video_assistant_pipeline import process_video
from src.nlp.intent_labels import INTENTS


# --------------------------------------------------
# Model paths
# --------------------------------------------------

NLP_MODEL_PATH = (
    "models/nlp/query_embedding_model_expanded.keras"
)

YOLO_MODEL_PATH = (
    "backup/Experiment_02_Backup/best.pt"
)

RISK_MODEL_PATH = (
    "models/risk/risk_ann.keras"
)

VIDEO_PATH = (
    "data/test/test_video.mp4"
)


# --------------------------------------------------
# Load models
# --------------------------------------------------

print("Loading models...")

nlp_model = load_model(
    NLP_MODEL_PATH,
    compile=False
)

yolo_model = YOLO(
    YOLO_MODEL_PATH
)

risk_model = load_model(
    RISK_MODEL_PATH,
    compile=False
)

# Dynamically read 13 classes from YOLO model
YOLO_CLASSES = yolo_model.names


def get_yolo_classes():
    """Retrieve dynamic 13 class mapping directly from the YOLO model."""
    return dict(yolo_model.names)


print("Models loaded successfully.")


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
# NLP: Query -> Intent
# --------------------------------------------------

def predict_intent(query):

    with KERAS_LOCK:
        prediction = nlp_model.predict(
            tf.constant(
                [query],
                dtype=tf.string
            ),
            verbose=0
        )[0]

    intent_id = int(
        prediction.argmax()
    )

    confidence = float(
        prediction[intent_id]
    )

    return (
        INTENTS[intent_id],
        confidence
    )


# --------------------------------------------------
# Risk ANN: Detection -> Risk
# --------------------------------------------------

def predict_risk(
    detection,
    movement="stationary",
    ahead=False
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
        prediction = risk_model.predict(
            tf.constant(
                [features],
                dtype=tf.float32
            ),
            verbose=0
        )[0]

    risk_id = int(
        prediction.argmax()
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
# Risk -> DQN decision
# --------------------------------------------------

def make_dqn_decision(
    detection,
    risk_id,
    intent,
    movement="stationary",
    recent_alert=0
):

    risk_score = (
        risk_id / 3.0
    )

    ahead = (
        1
        if detection["position"] == "center"
        else 0
    )

    with KERAS_LOCK:
        decision = get_dqn_action(
            risk=risk_score,
            confidence=detection["confidence"],
            movement=movement,
            ahead=ahead,
            intent=intent,
            recent_alert=recent_alert
        )

    return decision


# --------------------------------------------------
# Vision: Image -> YOLO detections
# --------------------------------------------------

def detect_objects(input_source, conf=0.25):

    with YOLO_LOCK:
        results = yolo_model.predict(
            input_source,
            conf=conf,
            verbose=False
        )

    detections = process_yolo_detections(
        results
    )

    return detections



# --------------------------------------------------
# Query-specific processing
# --------------------------------------------------

def process_query(
    query,
    image_path
):
    t_start = time.perf_counter()
    stage_latencies = {
        "camera_ms": 0.0,
        "yolo_ms": 0.0,
        "nlp_ms": 0.0,
        "spatial_ms": 0.0,
        "risk_ms": 0.0,
        "dqn_ms": 0.0,
        "voice_ms": 0.0,
        "total_ms": 0.0
    }

    # Camera / Image load check latency
    t_cam = time.perf_counter()
    if isinstance(image_path, str) and os.path.exists(image_path):
        pass
    stage_latencies["camera_ms"] = round((time.perf_counter() - t_cam) * 1000.0, 2)

    # ----------------------------------------------
    # Step 1: NLP
    # ----------------------------------------------

    t_nlp = time.perf_counter()
    intent, intent_confidence = predict_intent(
        query
    )
    stage_latencies["nlp_ms"] = round((time.perf_counter() - t_nlp) * 1000.0, 2)


    # ----------------------------------------------
    # Step 2: Select processing pipeline
    # ----------------------------------------------

    if intent == "REPEAT_RESPONSE":

        previous_response = get_last_response()

        if previous_response is None:
            response = (
                "There is no previous response "
                "to repeat."
            )
        else:
            response = previous_response

        detections = []
        movement_events = []


    elif intent == "MORE_DETAIL":

        previous_detections = get_last_detections()

        response = generate_more_detail_response(
            previous_detections
        )

        detections = previous_detections
        movement_events = []


    elif intent == "DETECT_APPROACHING":

        # Video pipeline:
        # YOLO -> Tracking -> LSTM ->
        # Risk ANN -> AlertManager -> DQN

        t_yolo = time.perf_counter()
        movement_events = process_video(
            VIDEO_PATH
        )
        stage_latencies["yolo_ms"] = round((time.perf_counter() - t_yolo) * 1000.0, 2)

        detections = []


    else:

        movement_events = []

        t_yolo = time.perf_counter()
        detections = detect_objects(
            image_path
        )
        stage_latencies["yolo_ms"] = round((time.perf_counter() - t_yolo) * 1000.0, 2)


    # ----------------------------------------------
    # Display basic information
    # ----------------------------------------------

    print("\nAI Vision Assistant")
    print("=" * 60)

    print(
        f"Query: {query}"
    )

    print(
        f"Intent: {intent}"
    )

    print(
        f"Intent confidence: "
        f"{intent_confidence:.4f}"
    )


    print("\nDetections")
    print("-" * 60)

    if not detections:

        print(
            "No known objects detected."
        )

    else:

        for detection in detections:

            print(
                f"{detection['object']} | "
                f"{detection['position']} | "
                f"{detection['confidence']:.2f}"
            )


    # ----------------------------------------------
    # Repeat previous response
    # ----------------------------------------------

    if intent == "REPEAT_RESPONSE":

        # Response already obtained
        # from conversation state.

        pass


    # ----------------------------------------------
    # More detail
    # ----------------------------------------------

    elif intent == "MORE_DETAIL":

        # Response already generated
        # from previous scene detections.

        pass


    # ----------------------------------------------
    # Detect approaching objects
    # ----------------------------------------------

    elif intent == "DETECT_APPROACHING":

        approaching_events = [
            event
            for event in movement_events
            if event["movement"] in [
                "Approaching",
                "Crossing"
            ]
        ]

        if not approaching_events:

            response = (
                "I don't detect anything "
                "approaching you."
            )

        else:

            event = max(
                approaching_events,
                key=lambda x:
                x["movement_confidence"]
            )

            response = (
                f"A {event['object']} is "
                f"{event['movement'].lower()} "
                f"from the "
                f"{event['position']}."
            )


    # ----------------------------------------------
    # Obstacle check
    # ----------------------------------------------

    elif intent == "OBSTACLE_CHECK":

        print("\nRisk Analysis")
        print("-" * 60)

        if not detections:

            response = (
                "I cannot detect any known "
                "objects ahead."
            )

        else:

            decisions = []

            for detection in detections:

                # Static-image assumption

                movement = "stationary"

                ahead = (
                    detection["position"]
                    == "center"
                )


                # Risk ANN

                (
                    risk_label,
                    risk_confidence,
                    risk_id
                ) = predict_risk(
                    detection,
                    movement=movement,
                    ahead=ahead
                )


                # DQN

                decision = make_dqn_decision(
                    detection=detection,
                    risk_id=risk_id,
                    intent=intent,
                    movement=movement,
                    recent_alert=0
                )


                print(
                    f"Object: "
                    f"{detection['object']}"
                )

                print(
                    f"Position: "
                    f"{detection['position']}"
                )

                print(
                    f"Risk: {risk_label} "
                    f"({risk_confidence:.4f})"
                )

                print(
                    f"DQN: "
                    f"{decision['action']}"
                )

                print(
                    "Q-values: "
                    f"{[round(q, 3) for q in decision['q_values']]}"
                )


                decisions.append({
                    "object":
                        detection["object"],

                    "position":
                        detection["position"],

                    "risk":
                        risk_label,

                    "risk_confidence":
                        risk_confidence,

                    "dqn_action":
                        decision["action"],

                    "q_values":
                        decision["q_values"]
                })


            # --------------------------------------
            # Final obstacle response
            # --------------------------------------

            if any(
                d["dqn_action"] == "WARN"
                for d in decisions
            ):

                response = (
                    "Warning. There may be "
                    "a safety risk ahead."
                )

            elif any(
                d["dqn_action"] == "SPEAK"
                for d in decisions
            ):

                response = (
                    "There is an object that "
                    "may require your attention."
                )

            else:

                response = (
                    "No immediate warning "
                    "is required."
                )


    # ----------------------------------------------
    # Scene summary
    # ----------------------------------------------

    elif intent == "SCENE_SUMMARY":

        response = summarize_scene(
            detections
        )



    elif intent == "COUNT_OBJECT":
        target = None

        object_aliases = {
            "people": "person",
            "persons": "person",
            "person": "person",

            "chairs": "chair",
            "chair": "chair",

            "tables": "table",
            "table": "table",

            "cars": "car",
            "car": "car",

            "bottles": "bottle",
            "bottle": "bottle",

            "backpacks": "backpack",
            "backpack": "backpack"
        }

        # Extract clean words from the query.
        import re
        words = re.findall(r"[a-z]+", query.lower())

        # Find the object being counted.
        for word in words:
            if word in object_aliases:
                target = object_aliases[word]
                break

        # If the user asks for all objects, count every detection.
        if "object" in words or "objects" in words:
            target = "all"

        if target == "all":
            count = len(detections)

            if count == 0:
                response = "I cannot see any objects."
            elif count == 1:
                response = "There is 1 object."
            else:
                response = f"There are {count} objects."

        elif target:
            matching_objects = [
                detection
                for detection in detections
                if detection["object"].lower() == target
            ]

            count = len(matching_objects)

            if target == "person":
                object_name = "person" if count == 1 else "people"
            else:
                object_name = target if count == 1 else target + "s"

            if count == 0:
                response = f"I cannot see any {object_name}."
            elif count == 1:
                response = f"There is 1 {object_name}."
            else:
                response = f"There are {count} {object_name}."

        else:
            response = "I could not determine which object you want me to count."

    # ----------------------------------------------
    # Objects in front
    # ----------------------------------------------

    elif intent == "FRONT_OBJECTS":

        front_objects = get_front_objects(
            detections
        )

        if not front_objects:

            response = (
                "I cannot detect any objects "
                "directly in front of you."
            )

        else:

            objects = [
                d["object"]
                for d in front_objects
            ]

            response = (
                "I can see "
                + " and ".join(objects)
                + " in front of you."
            )


    # ----------------------------------------------
    # Locate an object
    # ----------------------------------------------

    elif intent == "LOCATE_OBJECT":

        query_lower = query.lower()

        target = None

        for object_name in [
            "door",
            "chair",
            "table",
            "person",
            "car",
            "bottle",
            "backpack"
        ]:

            if object_name in query_lower:

                target = object_name
                break


        if target is None:

            response = (
                "I understood that you want "
                "to locate an object, but I "
                "could not identify which object."
            )

        else:

            locations = get_object_location(
                detections,
                target
            )

            if not locations:

                response = (
                    f"I cannot see a {target}."
                )

            else:

                position = locations[0]

                response = (
                    f"The {target} is in the "
                    f"{position}."
                )


    # ----------------------------------------------
    # Read text
    # ----------------------------------------------

    elif intent == "READ_TEXT":

        text_results = read_text(
            image_path
        )

        if not text_results:

            response = (
                "I cannot read any text "
                "in the image."
            )

        else:

            texts = [
                item["text"]
                for item in text_results
            ]

            response = (
                "The sign says "
                + ", ".join(texts)
                + "."
            )


    # ----------------------------------------------
    # Object identification
    # ----------------------------------------------

    elif intent == "OBJECT_IDENTIFICATION":

        identified_object = identify_object(
            detections
        )

        if identified_object is None:

            response = (
                "I cannot identify any "
                "known object."
            )

        else:

            object_name = (
                identified_object["object"]
            )

            response = (
                f"This appears to be a "
                f"{object_name}."
            )


    # ----------------------------------------------
    # Scene description
    # ----------------------------------------------

    elif intent == "SCENE_DESCRIPTION":

        if not detections:

            response = (
                "I cannot describe the scene "
                "because I cannot identify "
                "any objects."
            )

        else:

            object_counts = {}

            for detection in detections:

                object_name = detection[
                    "object"
                ]

                object_counts[object_name] = (
                    object_counts.get(
                        object_name,
                        0
                    )
                    + 1
                )


            scene = {
                "objects":
                    detections,

                "object_counts":
                    object_counts
            }


            response = (
                generate_scene_description_response(
                    scene
                )
            )


    # ----------------------------------------------
    # Other intents
    # ----------------------------------------------

    else:

        response = (
            f"The intent {intent} is recognized, "
            "but its processing module will be "
            "connected next."
        )


    # ----------------------------------------------
    # Final response
    # ----------------------------------------------

    print("\nResponse")
    print("-" * 60)

    print(response)


    # ----------------------------------------------
    # Save conversation state
    # ----------------------------------------------

    if intent not in [
        "REPEAT_RESPONSE",
        "MORE_DETAIL"
    ]:

        save_state(
            query=query,
            intent=intent,
            response=response,
            detections=detections,
            movement_events=movement_events
        )


    # ----------------------------------------------
    # Telemetry: Risk ANN, DQN & Latency Breakdown
    # ----------------------------------------------

    if 'decisions' in locals() and decisions:
        top_decision = next((d for d in decisions if d["dqn_action"] == "WARN"), None)
        if not top_decision:
            top_decision = next((d for d in decisions if d["dqn_action"] == "SPEAK"), None)
        if not top_decision:
            top_decision = decisions[0]
    elif detections:
        # Evaluate Risk ANN and Assistive DQN on the primary object in scene
        primary_det = next((d for d in detections if d.get("position") == "center"), detections[0])
        is_ahead = (primary_det.get("position") == "center")

        t_r = time.perf_counter()
        risk_label, risk_conf, risk_id = predict_risk(
            primary_det,
            movement="stationary",
            ahead=is_ahead
        )
        stage_latencies["risk_ms"] = round((time.perf_counter() - t_r) * 1000.0, 2)

        t_d = time.perf_counter()
        dqn_res = make_dqn_decision(
            detection=primary_det,
            risk_id=risk_id,
            intent=intent,
            movement="stationary",
            recent_alert=0
        )
        stage_latencies["dqn_ms"] = round((time.perf_counter() - t_d) * 1000.0, 2)

        top_decision = {
            "object": primary_det.get("object"),
            "position": primary_det.get("position"),
            "risk": risk_label,
            "risk_confidence": risk_conf,
            "dqn_action": dqn_res["action"],
            "q_values": dqn_res["q_values"]
        }
    else:
        top_decision = {
            "object": None,
            "position": None,
            "risk": "LOW",
            "risk_confidence": 1.0,
            "dqn_action": "WAIT",
            "q_values": [0.0, 0.0, 0.0]
        }

    # Derive real spatial reasoning latency from total elapsed minus model inference
    elapsed_so_far = (time.perf_counter() - t_start) * 1000.0
    stage_latencies["spatial_ms"] = round(max(0.0, elapsed_so_far - stage_latencies["yolo_ms"] - stage_latencies["nlp_ms"] - stage_latencies["risk_ms"] - stage_latencies["dqn_ms"] - stage_latencies["camera_ms"]), 2)
    stage_latencies["total_ms"] = round((time.perf_counter() - t_start) * 1000.0, 2)

    # ----------------------------------------------
    # Return structured result
    # ----------------------------------------------

    return {
        "query":
            query,

        "intent":
            intent,

        "intent_confidence":
            intent_confidence,

        "detections":
            detections,

        "movement_events":
            movement_events,

        "response":
            response,

        "risk":
            top_decision["risk"],

        "risk_confidence":
            top_decision.get("risk_confidence", 1.0),

        "dqn_action":
            top_decision["dqn_action"],

        "q_values":
            top_decision["q_values"],

        "stage_latencies":
            stage_latencies,

        "latency_ms":
            stage_latencies["total_ms"]
    }


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    scene_image = (
        "data/test/test_scene.jpg"
    )

    queries = [
        (
            "Where is the door?",
            "data/test/door_test.png"
        ),
        (
            "Repeat that.",
            "data/test/door_test.png"
        ),
        (
            "What is around me?",
            scene_image
        ),
        (
            "Give me more detail.",
            scene_image
        ),
        (
            "Read the text in this image.",
            scene_image
        ),
        (
            "Are there any obstacles ahead?",
            scene_image
        ),
        (
            "What objects are in front of me?",
            scene_image
        ),
        (
            "What is this object?",
            scene_image
        ),
        (
            "Describe the scene for me.",
            scene_image
        ),
        ("Is anything approaching me?", 
            scene_image
        )
    ]


    for query, image_path in queries:

        process_query(
            query,
            image_path
        )
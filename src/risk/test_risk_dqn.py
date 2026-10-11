import cv2
import numpy as np
from ultralytics import YOLO

from src.risk.risk_features import create_risk_features
from src.dqn.dqn_bridge import get_dqn_action


# --------------------------------------------------
# Paths
# --------------------------------------------------

YOLO_MODEL_PATH = "backup/Experiment_02_Backup/best.pt"
RISK_MODEL_PATH = "models/risk/risk_ann.keras"
TEST_IMAGE_PATH = "data/test/door_test.png"


# --------------------------------------------------
# Load models
# --------------------------------------------------

yolo_model = YOLO(YOLO_MODEL_PATH)

from tensorflow.keras.models import load_model

risk_model = load_model(
    RISK_MODEL_PATH,
    compile=False
)


# --------------------------------------------------
# Risk prediction
# --------------------------------------------------

def predict_risk(
    object_name,
    position,
    width,
    height,
    image_width,
    image_height,
    confidence,
    movement="stationary",
    ahead=False
):

    features = create_risk_features(
        object_name=object_name,
        position=position,
        width=width,
        height=height,
        image_width=image_width,
        image_height=image_height,
        confidence=confidence,
        movement=movement,
        ahead=ahead
    )

    features = np.array(
        [features],
        dtype=np.float32
    )

    probabilities = risk_model.predict(
        features,
        verbose=0
    )[0]

    risk_class = int(
        np.argmax(probabilities)
    )

    risk_names = [
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"
    ]

    risk_name = risk_names[risk_class]

    # Convert risk class to the 0–1 score
    # expected by the DQN.
    risk_score = risk_class / 3.0

    return {
        "risk_class": risk_class,
        "risk_name": risk_name,
        "risk_score": risk_score,
        "risk_confidence": float(
            probabilities[risk_class]
        ),
        "probabilities": probabilities.tolist()
    }


# --------------------------------------------------
# Main test
# --------------------------------------------------

image = cv2.imread(TEST_IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not read: {TEST_IMAGE_PATH}"
    )

image_height, image_width = image.shape[:2]

results = yolo_model(
    image,
    conf=0.25,
    verbose=False
)

print()
print("AI Vision Assistant")
print("YOLO → Risk ANN → DQN")
print("=" * 60)


for result in results:

    if result.boxes is None:
        continue

    for box in result.boxes:

        x1, y1, x2, y2 = (
            box.xyxy[0].tolist()
        )

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        object_name = result.names[class_id]

        width = x2 - x1
        height = y2 - y1

        x_center = (x1 + x2) / 2

        normalized_x = (
            x_center / image_width
        )

        if normalized_x < 0.33:
            position = "left"

        elif normalized_x < 0.66:
            position = "center"

        else:
            position = "right"

        ahead = position == "center"

        # ------------------------------------------
        # Risk ANN
        # ------------------------------------------

        risk = predict_risk(
            object_name=object_name,
            position=position,
            width=width,
            height=height,
            image_width=image_width,
            image_height=image_height,
            confidence=confidence,
            movement="stationary",
            ahead=ahead
        )

        # ------------------------------------------
        # DQN
        # ------------------------------------------

        decision = get_dqn_action(
            risk=risk["risk_score"],
            confidence=confidence,
            movement="stationary",
            ahead=int(ahead),
            intent="OBSTACLE_CHECK",
            recent_alert=0
        )

        print()
        print(f"Object: {object_name}")
        print(f"Position: {position}")
        print(f"Detection confidence: {confidence:.4f}")

        print(
            f"Risk: {risk['risk_name']} "
            f"({risk['risk_score']:.2f})"
        )

        print(
            f"Risk confidence: "
            f"{risk['risk_confidence']:.4f}"
        )

        print(
            f"DQN decision: "
            f"{decision['action']}"
        )

        print(
            f"DQN Q-values: "
            f"{decision['q_values']}"
        )


print()
print("Risk ANN + DQN integration test completed.")
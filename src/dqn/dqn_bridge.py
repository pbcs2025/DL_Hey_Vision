import numpy as np
from tensorflow.keras.models import load_model

from .assistive_environment import MOVEMENT_VALUES


# --------------------------------------------------
# DQN model
# --------------------------------------------------

DQN_MODEL_PATH = "models/dqn/assistive_dqn.keras"

dqn_model = load_model(
    DQN_MODEL_PATH,
    compile=False
)


# --------------------------------------------------
# Query priority
# --------------------------------------------------

QUERY_PRIORITIES = {
    "SCENE_SUMMARY": 0.4,
    "FRONT_OBJECTS": 0.7,
    "LOCATE_OBJECT": 0.8,
    "READ_TEXT": 0.8,
    "DETECT_APPROACHING": 1.0,
    "OBSTACLE_CHECK": 1.0,
    "OBJECT_IDENTIFICATION": 0.5,
    "SCENE_DESCRIPTION": 0.4,
    "REPEAT_RESPONSE": 0.3,
    "MORE_DETAIL": 0.5
}


# --------------------------------------------------
# Create DQN state
# --------------------------------------------------

def create_dqn_state(
    risk,
    confidence,
    movement,
    ahead,
    intent,
    recent_alert
):
    """
    Create the six-dimensional state expected
    by the trained DQN.

    State:
        1. risk
        2. confidence
        3. movement
        4. ahead
        5. priority
        6. recent_alert
    """

    movement = movement.lower().replace(" ", "_")

    movement_value = MOVEMENT_VALUES.get(
        movement,
        0.0
    )

    priority = QUERY_PRIORITIES.get(
        intent,
        0.5
    )

    state = np.array(
        [[
            risk,
            confidence,
            movement_value,
            ahead,
            priority,
            recent_alert
        ]],
        dtype=np.float32
    )

    return state


# --------------------------------------------------
# Select DQN action
# --------------------------------------------------

def get_dqn_action(
    risk,
    confidence,
    movement,
    ahead,
    intent,
    recent_alert
):
    """
    Use the trained DQN to select:

        WAIT
        SPEAK
        WARN
    """

    state = create_dqn_state(
        risk=risk,
        confidence=confidence,
        movement=movement,
        ahead=ahead,
        intent=intent,
        recent_alert=recent_alert
    )

    q_values = dqn_model.predict(
        state,
        verbose=0
    )[0]

    action = int(np.argmax(q_values))

    action_names = {
        0: "WAIT",
        1: "SPEAK",
        2: "WARN"
    }

    return {
        "action": action_names[action],
        "action_id": action,
        "state": state[0].tolist(),
        "q_values": q_values.tolist()
    }

if __name__ == "__main__":

    print("AI Vision Assistant - DQN Bridge Test")
    print("=" * 50)

    test_cases = [
        {
            "name": "Low-risk situation",
            "risk": 0.10,
            "confidence": 0.90,
            "movement": "stationary",
            "ahead": 0,
            "intent": "SCENE_SUMMARY",
            "recent_alert": 0
        },
        {
            "name": "Critical approaching obstacle",
            "risk": 0.90,
            "confidence": 0.95,
            "movement": "approaching",
            "ahead": 1,
            "intent": "OBSTACLE_CHECK",
            "recent_alert": 0
        },
        {
            "name": "Already alerted",
            "risk": 0.80,
            "confidence": 0.95,
            "movement": "stationary",
            "ahead": 1,
            "intent": "OBSTACLE_CHECK",
            "recent_alert": 1
        }
    ]

    for case in test_cases:

        result = get_dqn_action(
            risk=case["risk"],
            confidence=case["confidence"],
            movement=case["movement"],
            ahead=case["ahead"],
            intent=case["intent"],
            recent_alert=case["recent_alert"]
        )

        print(f"\nScenario: {case['name']}")
        print(f"Action: {result['action']}")
        print(f"State: {result['state']}")
        print(f"Q-values: {result['q_values']}")

import numpy as np
import tensorflow as tf

from .assistive_environment import AssistiveEnvironment


# ============================================================
# DQN EXPERIMENT 02 MODEL
# ============================================================

MODEL_PATH = "models/dqn/assistive_dqn_exp02.keras"


# ============================================================
# 24 CONTROLLED TEST SCENARIOS
# SAME SCENARIOS USED FOR EXPERIMENT 01
# ============================================================

scenarios = [

    # --------------------------------------------------------
    # LOW RISK
    # --------------------------------------------------------

    {
        "name": "Low risk stationary, no alert",
        "risk": 0.10,
        "confidence": 0.90,
        "movement": "stationary",
        "ahead": 0,
        "priority": 0.10,
        "recent_alert": 0,
        "expected": "WAIT"
    },

    {
        "name": "Low risk stationary ahead",
        "risk": 0.20,
        "confidence": 0.90,
        "movement": "stationary",
        "ahead": 1,
        "priority": 0.20,
        "recent_alert": 0,
        "expected": "WAIT"
    },

    {
        "name": "Low risk moving away",
        "risk": 0.15,
        "confidence": 0.85,
        "movement": "moving_away",
        "ahead": 0,
        "priority": 0.20,
        "recent_alert": 0,
        "expected": "WAIT"
    },

    {
        "name": "Low risk already alerted",
        "risk": 0.20,
        "confidence": 0.90,
        "movement": "stationary",
        "ahead": 0,
        "priority": 0.20,
        "recent_alert": 1,
        "expected": "WAIT"
    },


    # --------------------------------------------------------
    # HIGH PRIORITY INFORMATION
    # --------------------------------------------------------

    {
        "name": "High priority information",
        "risk": 0.40,
        "confidence": 0.90,
        "movement": "stationary",
        "ahead": 0,
        "priority": 0.90,
        "recent_alert": 0,
        "expected": "SPEAK"
    },

    {
        "name": "High priority object ahead",
        "risk": 0.50,
        "confidence": 0.90,
        "movement": "stationary",
        "ahead": 1,
        "priority": 0.80,
        "recent_alert": 0,
        "expected": "SPEAK"
    },

    {
        "name": "Medium risk high priority",
        "risk": 0.35,
        "confidence": 0.80,
        "movement": "moving_away",
        "ahead": 0,
        "priority": 0.75,
        "recent_alert": 0,
        "expected": "SPEAK"
    },

    {
        "name": "High priority after recent alert",
        "risk": 0.40,
        "confidence": 0.90,
        "movement": "stationary",
        "ahead": 0,
        "priority": 0.90,
        "recent_alert": 1,
        "expected": "WAIT"
    },


    # --------------------------------------------------------
    # APPROACHING
    # --------------------------------------------------------

    {
        "name": "Approaching ahead",
        "risk": 0.60,
        "confidence": 0.95,
        "movement": "approaching",
        "ahead": 1,
        "priority": 0.50,
        "recent_alert": 0,
        "expected": "WARN"
    },

    {
        "name": "Approaching ahead, high risk",
        "risk": 0.85,
        "confidence": 0.95,
        "movement": "approaching",
        "ahead": 1,
        "priority": 0.80,
        "recent_alert": 0,
        "expected": "WARN"
    },

    {
        "name": "Approaching not ahead",
        "risk": 0.60,
        "confidence": 0.90,
        "movement": "approaching",
        "ahead": 0,
        "priority": 0.50,
        "recent_alert": 0,
        "expected": "SPEAK"
    },

    {
        "name": "Approaching after previous alert",
        "risk": 0.80,
        "confidence": 0.95,
        "movement": "approaching",
        "ahead": 1,
        "priority": 0.80,
        "recent_alert": 1,
        "expected": "WARN"
    },


    # --------------------------------------------------------
    # CROSSING
    # --------------------------------------------------------

    {
        "name": "Crossing ahead",
        "risk": 0.60,
        "confidence": 0.90,
        "movement": "crossing",
        "ahead": 1,
        "priority": 0.60,
        "recent_alert": 0,
        "expected": "WARN"
    },

    {
        "name": "High risk crossing",
        "risk": 0.85,
        "confidence": 0.95,
        "movement": "crossing",
        "ahead": 1,
        "priority": 0.80,
        "recent_alert": 0,
        "expected": "WARN"
    },

    {
        "name": "Crossing not ahead",
        "risk": 0.50,
        "confidence": 0.90,
        "movement": "crossing",
        "ahead": 0,
        "priority": 0.50,
        "recent_alert": 0,
        "expected": "SPEAK"
    },

    {
        "name": "Crossing after previous alert",
        "risk": 0.80,
        "confidence": 0.95,
        "movement": "crossing",
        "ahead": 1,
        "priority": 0.80,
        "recent_alert": 1,
        "expected": "WARN"
    },


    # --------------------------------------------------------
    # CRITICAL STATIONARY
    # --------------------------------------------------------

    {
        "name": "Critical stationary, no alert",
        "risk": 0.80,
        "confidence": 0.95,
        "movement": "stationary",
        "ahead": 1,
        "priority": 0.80,
        "recent_alert": 0,
        "expected": "WARN"
    },

    {
        "name": "Critical stationary, already warned",
        "risk": 0.80,
        "confidence": 0.95,
        "movement": "stationary",
        "ahead": 1,
        "priority": 0.80,
        "recent_alert": 1,
        "expected": "WAIT"
    },

    {
        "name": "Critical stationary, low priority",
        "risk": 0.90,
        "confidence": 0.90,
        "movement": "stationary",
        "ahead": 1,
        "priority": 0.20,
        "recent_alert": 0,
        "expected": "WARN"
    },

    {
        "name": "Critical stationary not ahead",
        "risk": 0.90,
        "confidence": 0.90,
        "movement": "stationary",
        "ahead": 0,
        "priority": 0.80,
        "recent_alert": 0,
        "expected": "SPEAK"
    },


    # --------------------------------------------------------
    # MOVING AWAY
    # --------------------------------------------------------

    {
        "name": "Moving away, moderate risk",
        "risk": 0.40,
        "confidence": 0.90,
        "movement": "moving_away",
        "ahead": 1,
        "priority": 0.40,
        "recent_alert": 0,
        "expected": "SPEAK"
    },

    {
        "name": "Moving away, high risk",
        "risk": 0.80,
        "confidence": 0.90,
        "movement": "moving_away",
        "ahead": 1,
        "priority": 0.70,
        "recent_alert": 0,
        "expected": "SPEAK"
    },


    # --------------------------------------------------------
    # ALERT FATIGUE
    # --------------------------------------------------------

    {
        "name": "Moderate risk already alerted",
        "risk": 0.50,
        "confidence": 0.90,
        "movement": "stationary",
        "ahead": 1,
        "priority": 0.40,
        "recent_alert": 1,
        "expected": "WAIT"
    },

    {
        "name": "High risk moving away already alerted",
        "risk": 0.80,
        "confidence": 0.90,
        "movement": "moving_away",
        "ahead": 1,
        "priority": 0.40,
        "recent_alert": 1,
        "expected": "WAIT"
    }
]


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading DQN Experiment 02...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")
print("Model:", MODEL_PATH)


# ============================================================
# ENVIRONMENT
# ============================================================

environment = AssistiveEnvironment()


action_names = {
    0: "WAIT",
    1: "SPEAK",
    2: "WARN"
}


# ============================================================
# EVALUATION
# ============================================================

correct = 0

results = []


for i, scenario in enumerate(scenarios, start=1):

    # Convert scenario into state vector
    state = environment.reset(scenario)

    # Add batch dimension
    state_input = np.expand_dims(state, axis=0)

    # Get Q-values
    q_values = model.predict(
        state_input,
        verbose=0
    )[0]

    # Select action with highest Q-value
    selected_action = int(np.argmax(q_values))

    predicted = action_names[selected_action]

    expected = scenario["expected"]

    # Check correctness
    is_correct = predicted == expected

    if is_correct:
        correct += 1

    # Store result
    results.append({
        "scenario": scenario["name"],
        "expected": expected,
        "predicted": predicted,
        "correct": is_correct
    })

    # Print result
    print()
    print(f"{i:02d}. {scenario['name']}")

    print(f"    Expected : {expected}")
    print(f"    Predicted: {predicted}")

    print(
        f"    Q-values -> "
        f"WAIT={q_values[0]:.3f}, "
        f"SPEAK={q_values[1]:.3f}, "
        f"WARN={q_values[2]:.3f}"
    )

    if is_correct:
        print("    Result   : CORRECT")
    else:
        print("    Result   : INCORRECT")


# ============================================================
# OVERALL ACCURACY
# ============================================================

total = len(scenarios)

accuracy = correct / total


print()
print("=" * 70)
print("DQN EXPERIMENT 02 EVALUATION SUMMARY")
print("=" * 70)

print(f"Correct decisions : {correct}/{total}")
print(f"Decision accuracy : {accuracy * 100:.2f}%")


# ============================================================
# ACTION DISTRIBUTION
# ============================================================

print()
print("Predicted action distribution:")

for action in ["WAIT", "SPEAK", "WARN"]:

    count = sum(
        1
        for result in results
        if result["predicted"] == action
    )

    print(f"{action:6}: {count}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

actions = ["WAIT", "SPEAK", "WARN"]


print()
print("Confusion Matrix")
print("Rows = Expected | Columns = Predicted")
print()

print(
    f"{'':10}"
    f"{'WAIT':>8}"
    f"{'SPEAK':>8}"
    f"{'WARN':>8}"
)


for expected_action in actions:

    row = []

    for predicted_action in actions:

        count = sum(
            1
            for result in results
            if result["expected"] == expected_action
            and result["predicted"] == predicted_action
        )

        row.append(count)

    print(
        f"{expected_action:10}"
        f"{row[0]:8}"
        f"{row[1]:8}"
        f"{row[2]:8}"
    )


print("=" * 70)


import numpy as np
import tensorflow as tf

from .assistive_environment import AssistiveEnvironment


MODELS = {
    "Experiment 01": "models/dqn/assistive_dqn.keras",
    "Experiment 02": "models/dqn/assistive_dqn_exp02.keras"
}

scenarios = [

    # LOW RISK
    {
        "name": "Low risk stationary, no alert",
        "risk": 0.10, "confidence": 0.90,
        "movement": "stationary", "ahead": 0,
        "priority": 0.10, "recent_alert": 0,
        "expected": "WAIT"
    },
    {
        "name": "Low risk stationary ahead",
        "risk": 0.20, "confidence": 0.90,
        "movement": "stationary", "ahead": 1,
        "priority": 0.20, "recent_alert": 0,
        "expected": "WAIT"
    },
    {
        "name": "Low risk moving away",
        "risk": 0.15, "confidence": 0.85,
        "movement": "moving_away", "ahead": 0,
        "priority": 0.20, "recent_alert": 0,
        "expected": "WAIT"
    },
    {
        "name": "Low risk already alerted",
        "risk": 0.20, "confidence": 0.90,
        "movement": "stationary", "ahead": 0,
        "priority": 0.20, "recent_alert": 1,
        "expected": "WAIT"
    },

    # HIGH PRIORITY
    {
        "name": "High priority information",
        "risk": 0.40, "confidence": 0.90,
        "movement": "stationary", "ahead": 0,
        "priority": 0.90, "recent_alert": 0,
        "expected": "SPEAK"
    },
    {
        "name": "High priority object ahead",
        "risk": 0.50, "confidence": 0.90,
        "movement": "stationary", "ahead": 1,
        "priority": 0.80, "recent_alert": 0,
        "expected": "SPEAK"
    },
    {
        "name": "Medium risk high priority",
        "risk": 0.35, "confidence": 0.80,
        "movement": "moving_away", "ahead": 0,
        "priority": 0.75, "recent_alert": 0,
        "expected": "SPEAK"
    },
    {
        "name": "High priority after recent alert",
        "risk": 0.40, "confidence": 0.90,
        "movement": "stationary", "ahead": 0,
        "priority": 0.90, "recent_alert": 1,
        "expected": "WAIT"
    },

    # APPROACHING
    {
        "name": "Approaching ahead",
        "risk": 0.60, "confidence": 0.95,
        "movement": "approaching", "ahead": 1,
        "priority": 0.50, "recent_alert": 0,
        "expected": "WARN"
    },
    {
        "name": "Approaching ahead, high risk",
        "risk": 0.85, "confidence": 0.95,
        "movement": "approaching", "ahead": 1,
        "priority": 0.80, "recent_alert": 0,
        "expected": "WARN"
    },
    {
        "name": "Approaching not ahead",
        "risk": 0.60, "confidence": 0.90,
        "movement": "approaching", "ahead": 0,
        "priority": 0.50, "recent_alert": 0,
        "expected": "SPEAK"
    },
    {
        "name": "Approaching after previous alert",
        "risk": 0.80, "confidence": 0.95,
        "movement": "approaching", "ahead": 1,
        "priority": 0.80, "recent_alert": 1,
        "expected": "WARN"
    },

    # CROSSING
    {
        "name": "Crossing ahead",
        "risk": 0.60, "confidence": 0.90,
        "movement": "crossing", "ahead": 1,
        "priority": 0.60, "recent_alert": 0,
        "expected": "WARN"
    },
    {
        "name": "High risk crossing",
        "risk": 0.85, "confidence": 0.95,
        "movement": "crossing", "ahead": 1,
        "priority": 0.80, "recent_alert": 0,
        "expected": "WARN"
    },
    {
        "name": "Crossing not ahead",
        "risk": 0.50, "confidence": 0.90,
        "movement": "crossing", "ahead": 0,
        "priority": 0.50, "recent_alert": 0,
        "expected": "SPEAK"
    },
    {
        "name": "Crossing after previous alert",
        "risk": 0.80, "confidence": 0.95,
        "movement": "crossing", "ahead": 1,
        "priority": 0.80, "recent_alert": 1,
        "expected": "WARN"
    },

    # CRITICAL STATIONARY
    {
        "name": "Critical stationary, no alert",
        "risk": 0.80, "confidence": 0.95,
        "movement": "stationary", "ahead": 1,
        "priority": 0.80, "recent_alert": 0,
        "expected": "WARN"
    },
    {
        "name": "Critical stationary, already warned",
        "risk": 0.80, "confidence": 0.95,
        "movement": "stationary", "ahead": 1,
        "priority": 0.80, "recent_alert": 1,
        "expected": "WAIT"
    },
    {
        "name": "Critical stationary, low priority",
        "risk": 0.90, "confidence": 0.90,
        "movement": "stationary", "ahead": 1,
        "priority": 0.20, "recent_alert": 0,
        "expected": "WARN"
    },
    {
        "name": "Critical stationary not ahead",
        "risk": 0.90, "confidence": 0.90,
        "movement": "stationary", "ahead": 0,
        "priority": 0.80, "recent_alert": 0,
        "expected": "SPEAK"
    },

    # MOVING AWAY
    {
        "name": "Moving away, moderate risk",
        "risk": 0.40, "confidence": 0.90,
        "movement": "moving_away", "ahead": 1,
        "priority": 0.40, "recent_alert": 0,
        "expected": "SPEAK"
    },
    {
        "name": "Moving away, high risk",
        "risk": 0.80, "confidence": 0.90,
        "movement": "moving_away", "ahead": 1,
        "priority": 0.70, "recent_alert": 0,
        "expected": "SPEAK"
    },

    # ALERT FATIGUE
    {
        "name": "Moderate risk already alerted",
        "risk": 0.50, "confidence": 0.90,
        "movement": "stationary", "ahead": 1,
        "priority": 0.40, "recent_alert": 1,
        "expected": "WAIT"
    },
    {
        "name": "High risk moving away already alerted",
        "risk": 0.80, "confidence": 0.90,
        "movement": "moving_away", "ahead": 1,
        "priority": 0.40, "recent_alert": 1,
        "expected": "WAIT"
    }
]


action_names = {
    0: "WAIT",
    1: "SPEAK",
    2: "WARN"
}


environment = AssistiveEnvironment()

all_results = {}


for model_name, model_path in MODELS.items():

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    model = tf.keras.models.load_model(model_path)

    results = []

    for i, scenario in enumerate(scenarios, start=1):

        state = environment.reset(scenario)

        q_values = model.predict(
            np.expand_dims(state, axis=0),
            verbose=0
        )[0]

        action = int(np.argmax(q_values))

        predicted = action_names[action]
        expected = scenario["expected"]

        correct = predicted == expected

        results.append({
            "expected": expected,
            "predicted": predicted,
            "correct": correct
        })

        print(
            f"{i:02d}. "
            f"{scenario['name']:<48} "
            f"Expected: {expected:<6} "
            f"Predicted: {predicted:<6} "
            f"{'OK' if correct else 'WRONG'}"
        )

    all_results[model_name] = results

    total_correct = sum(r["correct"] for r in results)
    accuracy = total_correct / len(results)

    print("\nAccuracy:", f"{accuracy * 100:.2f}%")

    print("\nPer-action accuracy:")

    for action in ["WAIT", "SPEAK", "WARN"]:

        action_results = [
            r for r in results
            if r["expected"] == action
        ]

        action_correct = sum(
            r["correct"] for r in action_results
        )

        action_accuracy = (
            action_correct / len(action_results)
            if action_results else 0
        )

        print(
            f"{action:6}: "
            f"{action_correct}/{len(action_results)} "
            f"({action_accuracy * 100:.2f}%)"
        )


# --------------------------------------------------
# Direct comparison
# --------------------------------------------------

print("\n" + "=" * 70)
print("EXPERIMENT COMPARISON")
print("=" * 70)

for model_name, results in all_results.items():

    correct = sum(r["correct"] for r in results)
    accuracy = correct / len(results)

    print(
        f"{model_name}: "
        f"{correct}/{len(results)} "
        f"({accuracy * 100:.2f}%)"
    )

exp01 = all_results["Experiment 01"]
exp02 = all_results["Experiment 02"]

exp01_accuracy = sum(r["correct"] for r in exp01) / len(exp01)
exp02_accuracy = sum(r["correct"] for r in exp02) / len(exp02)

print(
    "\nAccuracy change: "
    f"{(exp02_accuracy - exp01_accuracy) * 100:+.2f} percentage points"
)

print("=" * 70)

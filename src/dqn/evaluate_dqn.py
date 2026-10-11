
import numpy as np
import tensorflow as tf

from .assistive_environment import AssistiveEnvironment

MODEL_PATH = "models/dqn/assistive_dqn.keras"

scenarios = [
    {
        "name": "Critical approaching person ahead",
        "risk": 0.90,
        "confidence": 0.95,
        "movement": "approaching",
        "ahead": 1,
        "priority": 0.90,
        "recent_alert": 0,
        "expected": "WARN"
    },
    {
        "name": "Low-risk stationary object",
        "risk": 0.10,
        "confidence": 0.90,
        "movement": "stationary",
        "ahead": 0,
        "priority": 0.10,
        "recent_alert": 0,
        "expected": "WAIT"
    },
    {
        "name": "High-priority information",
        "risk": 0.40,
        "confidence": 0.90,
        "movement": "stationary",
        "ahead": 0,
        "priority": 0.90,
        "recent_alert": 0,
        "expected": "SPEAK"
    },
    {
        "name": "Critical stationary situation already warned",
        "risk": 0.80,
        "confidence": 0.95,
        "movement": "stationary",
        "ahead": 1,
        "priority": 0.80,
        "recent_alert": 1,
        "expected": "WAIT"
    },
    {
        "name": "Critical approaching situation after earlier alert",
        "risk": 0.80,
        "confidence": 0.95,
        "movement": "approaching",
        "ahead": 1,
        "priority": 0.80,
        "recent_alert": 1,
        "expected": "WARN"
    }
]

print("Loading trained DQN...")
model = tf.keras.models.load_model(MODEL_PATH)

environment = AssistiveEnvironment()

action_names = {
    0: "WAIT",
    1: "SPEAK",
    2: "WARN"
}

correct = 0

print("\nDQN Evaluation")
print("=" * 70)

for scenario in scenarios:

    state = environment.reset(scenario)

    q_values = model.predict(
        np.expand_dims(state, axis=0),
        verbose=0
    )[0]

    selected_action = int(np.argmax(q_values))

    predicted_name = action_names[selected_action]
    expected_name = scenario["expected"]

    is_correct = predicted_name == expected_name

    if is_correct:
        correct += 1

    print(f"\nScenario: {scenario['name']}")
    print(f"State: {state}")

    print(
        f"Q(WAIT)  = {q_values[0]:8.3f}\n"
        f"Q(SPEAK) = {q_values[1]:8.3f}\n"
        f"Q(WARN)  = {q_values[2]:8.3f}"
    )

    print(f"Predicted action: {predicted_name}")
    print(f"Expected action:  {expected_name}")
    print(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}")

accuracy = correct / len(scenarios)

print("\n" + "=" * 70)
print(f"Correct decisions: {correct}/{len(scenarios)}")
print(f"Decision accuracy: {accuracy * 100:.2f}%")
print("=" * 70)

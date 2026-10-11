from src.dqn.dqn_bridge import get_dqn_action


test_cases = [
    {
        "name": "Low risk stationary object",
        "risk": 0.10,
        "confidence": 0.90,
        "movement": "Stationary",
        "ahead": 0,
        "intent": "SCENE_SUMMARY",
        "recent_alert": 0,
        "expected": "WAIT"
    },
    {
        "name": "Medium risk important query",
        "risk": 0.50,
        "confidence": 0.90,
        "movement": "Stationary",
        "ahead": 0,
        "intent": "FRONT_OBJECTS",
        "recent_alert": 0,
        "expected": "SPEAK"
    },
    {
        "name": "Critical approaching object ahead",
        "risk": 0.90,
        "confidence": 0.95,
        "movement": "Approaching",
        "ahead": 1,
        "intent": "OBSTACLE_CHECK",
        "recent_alert": 0,
        "expected": "WARN"
    },
    {
        "name": "High risk crossing object ahead",
        "risk": 0.80,
        "confidence": 0.90,
        "movement": "Crossing",
        "ahead": 1,
        "intent": "DETECT_APPROACHING",
        "recent_alert": 0,
        "expected": "WARN"
    },
    {
        "name": "Already alerted critical object",
        "risk": 0.90,
        "confidence": 0.95,
        "movement": "Approaching",
        "ahead": 1,
        "intent": "OBSTACLE_CHECK",
        "recent_alert": 1,
        "expected": "WAIT"
    },
    {
        "name": "Normal object identification",
        "risk": 0.30,
        "confidence": 0.90,
        "movement": "Stationary",
        "ahead": 0,
        "intent": "OBJECT_IDENTIFICATION",
        "recent_alert": 0,
        "expected": "SPEAK"
    }
]


print("=" * 70)
print("DQN DECISION EVALUATION")
print("=" * 70)

correct = 0

for test in test_cases:

    result = get_dqn_action(
        risk=test["risk"],
        confidence=test["confidence"],
        movement=test["movement"],
        ahead=test["ahead"],
        intent=test["intent"],
        recent_alert=test["recent_alert"]
    )

    actual = result["action"]
    expected = test["expected"]

    if actual == expected:
        status = "PASS"
        correct += 1
    else:
        status = "FAIL"

    print("\n" + "-" * 70)
    print(f"Scenario : {test['name']}")
    print(f"Expected : {expected}")
    print(f"Actual   : {actual}")
    print(f"Status   : {status}")
    print(f"Q-values : {[round(q, 4) for q in result['q_values']]}")

accuracy = correct / len(test_cases) * 100

print("\n" + "=" * 70)
print(f"DQN Evaluation Accuracy: {accuracy:.2f}%")
print(f"Passed: {correct}/{len(test_cases)}")
print("=" * 70)
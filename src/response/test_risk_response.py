from .risk_response import generate_risk_response


test_cases = [
    ("chair", "on your left", "LOW"),
    ("chair", "in front of you", "MEDIUM"),
    ("table", "in front of you", "HIGH"),
    ("person", "directly ahead", "CRITICAL")
]


print("\nAI Vision Assistant - Risk Response Test")
print("=" * 60)

for object_name, position, risk_level in test_cases:

    response = generate_risk_response(
        object_name,
        position,
        risk_level
    )

    print(f"\nObject: {object_name}")
    print(f"Position: {position}")
    print(f"Risk: {risk_level}")
    print(f"Assistant: {response}")
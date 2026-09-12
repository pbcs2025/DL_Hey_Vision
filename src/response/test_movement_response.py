from .movement_response import generate_movement_response


tests = [
    ("Approaching", 0.95, "left"),
    ("Approaching", 0.95, "center"),
    ("Approaching", 0.95, "right"),
    ("Approaching", 0.65, "left"),
    ("Stationary", 0.95, "center"),
    ("Moving Away", 0.95, "right"),
    ("Crossing", 0.90, "left"),
    ("Crossing", 0.90, "center"),
    ("Crossing", 0.90, "right"),
]


print("\nMovement Response Test")
print("=" * 50)


for movement, confidence, position in tests:

    response = generate_movement_response(
        movement,
        confidence,
        position
    )

    print(f"\nMovement:   {movement}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Position:   {position}")
    print(f"Response:   {response}")
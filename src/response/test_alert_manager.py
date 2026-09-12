from .alert_manager import AlertManager


manager = AlertManager()


tests = [
    (1, "Stationary", 0.95),
    (1, "Approaching", 0.95),
    (1, "Approaching", 0.92),
    (1, "Approaching", 0.98),
    (1, "Moving Away", 0.95),
    (1, "Moving Away", 0.90),
    (1, "Approaching", 0.94),
    (2, "Approaching", 0.91),
    (2, "Approaching", 0.93),
    (2, "Stationary", 0.96),
    (2, "Crossing", 0.90),
]


print("\nState-Based Alert Manager Test")
print("=" * 50)


for track_id, movement, confidence in tests:

    alert = manager.should_alert(
        track_id,
        movement,
        confidence
    )

    manager.update_state(
        track_id,
        movement
    )

    print(
        f"Track {track_id} | "
        f"{movement:12} | "
        f"Confidence: {confidence:.2f} | "
        f"Alert: {alert}"
    )
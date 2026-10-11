from src.temporal.alert_manager import AlertManager


alert_manager = AlertManager()


track_id = 4


test_sequence = [
    ("Approaching", 0.95),
    ("Approaching", 0.96),
    ("Approaching", 0.97),
    ("Approaching", 0.98),
    ("Stationary", 0.91),
    ("Approaching", 0.96),
    ("Approaching", 0.97),
    ("Crossing", 0.94),
]


print("AlertManager Test")
print("=" * 60)


for movement, confidence in test_sequence:

    alert = alert_manager.should_alert(
        track_id=track_id,
        movement=movement,
        confidence=confidence
    )

    print(
        f"Movement: {movement:12s} | "
        f"Confidence: {confidence:.2f} | "
        f"Alert: {alert}"
    )

    alert_manager.update_state(
        track_id,
        movement
    )
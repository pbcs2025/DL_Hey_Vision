from .track_history import TrackHistory


tracker = TrackHistory(sequence_length=30)

print("\nTrack History Test")
print("=" * 40)

for frame in range(35):

    features = [
        0.50,
        0.55,
        0.20 + frame * 0.01,
        0.30 + frame * 0.01
    ]

    tracker.update(track_id=1, features=features)

    sequence = tracker.get_sequence(track_id=1)

    if sequence is not None:
        print(
            f"Frame {frame + 1}: "
            f"sequence ready → {len(sequence)} frames"
        )
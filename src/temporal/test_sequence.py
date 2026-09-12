from .feature_extraction import build_sequence


# Simulated features from 10 consecutive frames
feature_history = [
    [0.50, 0.55, 0.15, 0.30],
    [0.50, 0.55, 0.18, 0.35],
    [0.50, 0.56, 0.21, 0.40],
    [0.51, 0.56, 0.24, 0.45],
    [0.51, 0.57, 0.27, 0.50],
    [0.51, 0.57, 0.30, 0.55],
    [0.52, 0.58, 0.33, 0.60],
    [0.52, 0.58, 0.36, 0.65],
    [0.52, 0.59, 0.39, 0.70],
    [0.53, 0.59, 0.42, 0.75]
]


sequence = build_sequence(
    feature_history,
    sequence_length=10
)


print("\nTemporal Sequence")
print("=" * 40)

for i, features in enumerate(sequence, start=1):
    print(f"Frame {i}: {features}")

print("\nSequence shape:")
print(f"({len(sequence)}, {len(sequence[0])})")
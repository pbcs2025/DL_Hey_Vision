import numpy as np


SEQUENCE_LENGTH = 30


def generate_stationary_sequence():
    sequence = []

    base_width = np.random.uniform(0.20, 0.35)
    base_height = np.random.uniform(0.35, 0.55)
    base_x = np.random.uniform(0.35, 0.65)
    base_y = np.random.uniform(0.45, 0.65)

    for _ in range(SEQUENCE_LENGTH):
        x = base_x + np.random.normal(0, 0.015)
        y = base_y + np.random.normal(0, 0.015)

        width = base_width + np.random.normal(0, 0.015)
        height = base_height + np.random.normal(0, 0.020)

        sequence.append([x, y, width, height])

    return sequence


def generate_approaching_sequence():
    sequence = []

    start_width = np.random.uniform(0.12, 0.22)
    start_height = np.random.uniform(0.25, 0.40)

    for i in range(SEQUENCE_LENGTH):

        progress = i / (SEQUENCE_LENGTH - 1)

        x = 0.50 + np.random.normal(0, 0.015)
        y = 0.55 + np.random.normal(0, 0.015)

        width = start_width + (0.25 * progress)
        height = start_height + (0.40 * progress)

        # Add measurement noise
        width += np.random.normal(0, 0.015)
        height += np.random.normal(0, 0.020)

        sequence.append([x, y, width, height])

    return sequence


def generate_moving_away_sequence():
    sequence = []

    start_width = np.random.uniform(0.35, 0.50)
    start_height = np.random.uniform(0.60, 0.80)

    for i in range(SEQUENCE_LENGTH):

        progress = i / (SEQUENCE_LENGTH - 1)

        x = 0.50 + np.random.normal(0, 0.015)
        y = 0.55 + np.random.normal(0, 0.015)

        width = start_width - (0.25 * progress)
        height = start_height - (0.40 * progress)

        width += np.random.normal(0, 0.015)
        height += np.random.normal(0, 0.020)

        sequence.append([x, y, width, height])

    return sequence


def generate_crossing_sequence():
    sequence = []

    start_x = np.random.uniform(0.10, 0.30)

    for i in range(SEQUENCE_LENGTH):

        progress = i / (SEQUENCE_LENGTH - 1)

        # Object moves horizontally across the scene
        x = start_x + (0.60 * progress)
        x += np.random.normal(0, 0.015)

        y = 0.55 + np.random.normal(0, 0.015)

        # Size remains approximately constant
        width = 0.25 + np.random.normal(0, 0.015)
        height = 0.45 + np.random.normal(0, 0.020)

        sequence.append([x, y, width, height])

    return sequence


def create_dataset(samples_per_class=250):

    X = []
    y = []

    generators = [
        (generate_stationary_sequence, 0),
        (generate_approaching_sequence, 1),
        (generate_moving_away_sequence, 2),
        (generate_crossing_sequence, 3)
    ]

    for generator, label in generators:

        for _ in range(samples_per_class):
            X.append(generator())
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    # Shuffle dataset
    indices = np.random.permutation(len(X))

    X = X[indices]
    y = y[indices]

    return X, y


if __name__ == "__main__":

    X, y = create_dataset(250)

    print("Dataset created")
    print("=" * 40)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("\nClass distribution:")
    print("Stationary:", np.sum(y == 0))
    print("Approaching:", np.sum(y == 1))
    print("Moving away:", np.sum(y == 2))
    print("Crossing:", np.sum(y == 3))
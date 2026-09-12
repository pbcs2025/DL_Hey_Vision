def extract_temporal_features(detection, image_width, image_height):
    """
    Extract normalized numerical features from
    one object detection.

    Features:
        x_center
        y_center
        width
        height

    All values are normalized to approximately 0–1.
    """

    x_center = detection["x_center"] / image_width
    y_center = detection["y_center"] / image_height
    width = detection["width"] / image_width
    height = detection["height"] / image_height

    return [
        x_center,
        y_center,
        width,
        height
    ]

def build_sequence(feature_history, sequence_length=10):
    """
    Build a fixed-length temporal sequence
    from a history of object features.

    Each feature vector contains:
        [x_center, y_center, width, height]

    Returns:
        A list containing the most recent
        sequence_length feature vectors.
    """

    if len(feature_history) < sequence_length:
        return None

    return feature_history[-sequence_length:]
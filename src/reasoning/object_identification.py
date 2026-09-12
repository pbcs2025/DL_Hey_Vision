def identify_object(detections):
    """
    Identify the most confidently detected object.

    The detection with the highest confidence
    is treated as the main object.
    """

    if not detections:
        return None

    best_detection = max(
        detections,
        key=lambda detection: detection["confidence"]
    )

    return best_detection
def is_potential_obstacle(detection):
    """
    Determine whether a detected object could
    be an obstacle in front of the user.
    """

    obstacle_objects = [
        "chair",
        "table",
        "person",
        "car",
        "bicycle",
        "motorcycle",
        "bus",
        "truck",
        "backpack",
        "door"
    ]

    return detection["object"] in obstacle_objects


def find_obstacles_ahead(detections):
    """
    Find potential obstacles located in the
    center/front region of the image.
    """

    obstacles = []

    for detection in detections:

        if detection["position"] != "center":
            continue

        if is_potential_obstacle(detection):
            obstacles.append(detection)

    return obstacles
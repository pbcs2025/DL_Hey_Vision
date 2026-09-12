def get_front_objects(detections):
    """
    Find objects located in the central region
    of the image.

    For now, we use the CENTER position as a
    2D approximation of 'in front'.
    """

    front_objects = []

    for detection in detections:

        if detection["position"] == "center":
            front_objects.append(detection)

    return front_objects


def get_objects_by_direction(detections, direction):
    """
    Return objects detected in a specific direction.

    direction can be:
        left
        center
        right
    """

    objects = []

    for detection in detections:

        if detection["position"] == direction:
            objects.append(detection)

    return objects
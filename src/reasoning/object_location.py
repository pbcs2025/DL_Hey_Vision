def find_objects_by_name(detections, object_name):
    """
    Find all detections matching a specific object name.
    """
    matching_objects = []

    for detection in detections:
        if detection["object"] == object_name:
            matching_objects.append(detection)

    return matching_objects


def get_object_location(detections, object_name):
    """
    Find the location of a specific object.

    Returns the detected object positions.
    """
    matching_objects = find_objects_by_name(
        detections,
        object_name
    )

    locations = []

    for detection in matching_objects:
        locations.append(detection["position"])

    return locations
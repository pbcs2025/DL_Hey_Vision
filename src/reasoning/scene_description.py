def create_scene_description(detections):
    """
    Create a structured description of the scene
    from detected objects.
    """

    if not detections:
        return {
            "objects": [],
            "object_counts": {},
            "total_objects": 0
        }

    object_counts = {}

    for detection in detections:
        object_name = detection["object"]

        if object_name not in object_counts:
            object_counts[object_name] = 0

        object_counts[object_name] += 1

    return {
        "objects": list(object_counts.keys()),
        "object_counts": object_counts,
        "total_objects": len(detections)
    }
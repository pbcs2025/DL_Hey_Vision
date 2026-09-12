def generate_scene_response(detections):
    """
    Generate a concise description of
    detected objects in the scene.
    """

    if not detections:
        return "I cannot identify any objects around you."

    object_counts = {}

    for detection in detections:

        object_name = detection["object"]

        if object_name not in object_counts:
            object_counts[object_name] = 0

        object_counts[object_name] += 1

    # Handle irregular plural forms
    plural_names = {
        "person": "people"
    }

    descriptions = []

    for object_name, count in object_counts.items():

        if count == 1:
            descriptions.append(
                f"a {object_name}"
            )

        else:
            plural_name = plural_names.get(
                object_name,
                object_name + "s"
            )

            descriptions.append(
                f"{count} {plural_name}"
            )

    if len(descriptions) == 1:

        objects_text = descriptions[0]

    elif len(descriptions) == 2:

        objects_text = (
            f"{descriptions[0]} and "
            f"{descriptions[1]}"
        )

    else:

        objects_text = (
            ", ".join(descriptions[:-1])
            + f", and {descriptions[-1]}"
        )

    return f"I can see {objects_text} around you."
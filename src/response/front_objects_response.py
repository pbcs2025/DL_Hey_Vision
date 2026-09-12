def generate_front_objects_response(front_objects):
    if not front_objects:
        return "I cannot identify anything directly in front of you."

    object_counts = {}

    for detection in front_objects:
        object_name = detection["object"]

        if object_name not in object_counts:
            object_counts[object_name] = 0

        object_counts[object_name] += 1

    descriptions = []

    for object_name, count in object_counts.items():

        if count == 1:
            descriptions.append(f"a {object_name}")
        else:
            descriptions.append(f"{count} {object_name}s")

    if len(descriptions) == 1:
        objects_text = descriptions[0]

    elif len(descriptions) == 2:
        objects_text = (
            f"{descriptions[0]} and {descriptions[1]}"
        )

    else:
        objects_text = (
            ", ".join(descriptions[:-1])
            + f", and {descriptions[-1]}"
        )

    return f"I can see {objects_text} in front of you."
def generate_scene_description_response(scene):
    """
    Generate a concise natural-language description
    from the structured scene representation.
    """

    if not scene or not scene["objects"]:
        return "I cannot describe the scene because I cannot identify any objects."

    object_counts = scene["object_counts"]

    plural_names = {
        "person": "people"
    }

    descriptions = []

    for object_name, count in object_counts.items():

        if count == 1:
            descriptions.append(f"a {object_name}")

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

    return f"The scene contains {objects_text}."
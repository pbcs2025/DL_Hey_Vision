def generate_obstacle_response(obstacles):
    """
    Generate a response for potential obstacles
    detected in front of the user.
    """

    if not obstacles:
        return "I do not detect an obstacle directly ahead."

    object_counts = {}

    for obstacle in obstacles:

        object_name = obstacle["object"]

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
            f"{descriptions[0]} and "
            f"{descriptions[1]}"
        )

    else:
        objects_text = (
            ", ".join(descriptions[:-1])
            + f", and {descriptions[-1]}"
        )


    return (
        f"Yes, I detect {objects_text} "
        f"directly ahead."
    )
def generate_more_detail_response(detections):
    """
    Generate a more detailed response using
    the previously detected scene.
    """

    if not detections:
        return (
            "I do not have enough information "
            "from the previous scene to provide "
            "more detail."
        )

    # Group detections by object type
    objects = {}

    for detection in detections:
        object_name = detection["object"]
        position = detection["position"]

        if object_name not in objects:
            objects[object_name] = []

        objects[object_name].append(position)

    descriptions = []

    for object_name, positions in objects.items():

        count = len(positions)

        # Count objects at each position
        position_counts = {}

        for position in positions:
            position_counts[position] = (
                position_counts.get(position, 0) + 1
            )

        # Sort positions by number of objects
        sorted_positions = sorted(
            position_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # One object
        if count == 1:
            position = positions[0]

            if position == "center":
                descriptions.append(
                    f"There is one {object_name} "
                    f"in the center."
                )
            else:
                descriptions.append(
                    f"There is one {object_name} "
                    f"on your {position}."
                )

        # Multiple objects
        else:
            position_parts = []

            for position, position_count in sorted_positions:

                if position == "center":
                    location = "in the center"
                else:
                    location = f"on your {position}"

                if position_count == 1:
                    position_parts.append(
                        f"one {location}"
                    )
                else:
                    position_parts.append(
                        f"{position_count} {location}"
                    )

            if len(position_parts) == 1:
                position_text = position_parts[0]

            elif len(position_parts) == 2:
                position_text = (
                    f"{position_parts[0]} and "
                    f"{position_parts[1]}"
                )

            else:
                position_text = (
                    ", ".join(position_parts[:-1])
                    + ", and "
                    + position_parts[-1]
                )

            descriptions.append(
                f"There are {count} {object_name}s: "
                f"{position_text}."
            )

    return (
        "Here is a more detailed view of the scene. "
        + " ".join(descriptions)
    )
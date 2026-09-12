from collections import defaultdict


# Natural names for objects
OBJECT_NAMES = {
    "person": ("person", "people"),
    "chair": ("chair", "chairs"),
    "table": ("table", "tables"),
    "car": ("car", "cars"),
    "bicycle": ("bicycle", "bicycles"),
    "motorcycle": ("motorcycle", "motorcycles"),
    "bus": ("bus", "buses"),
    "truck": ("truck", "trucks"),
    "backpack": ("backpack", "backpacks"),
    "bottle": ("bottle", "bottles"),
    "door": ("door", "doors"),
    "traffic_light": ("traffic light", "traffic lights"),
    "stop_sign": ("stop sign", "stop signs"),
}


def get_object_name(object_name, count):
    """
    Return the correct singular or plural name.
    """

    singular, plural = OBJECT_NAMES.get(
        object_name,
        (object_name, object_name + "s")
    )

    return singular if count == 1 else plural


def summarize_scene(detections):
    """
    Convert spatial detections into a concise,
    human-readable scene description.
    """

    if not detections:
        return "I cannot identify any objects around you."

    objects = defaultdict(list)

    # Group detections by object type
    for detection in detections:
        objects[detection["object"]].append(
            detection["position"]
        )

    sentences = []

    # Handle each detected object type
    for object_name, positions in objects.items():

        count = len(positions)

        # People are described individually by side
        if object_name == "person":

            if count == 1:
                sentences.append(
                    f"I can see a person on your {positions[0]}."
                )

            else:
                side_descriptions = []

                for position in dict.fromkeys(positions):
                    side_descriptions.append(
                        f"one on your {position}"
                    )

                if len(side_descriptions) == 2:
                    description = " and ".join(
                        side_descriptions
                    )
                else:
                    description = ", ".join(
                        side_descriptions
                    )

                sentences.append(
                    f"I can see {count} people, {description}."
                )

        # Many chairs are grouped as "several"
        elif object_name == "chair" and count >= 4:

            positions_text = format_positions(
                positions
            )

            sentences.append(
                f"There are several chairs "
                f"{positions_text}."
            )

        # Normal objects
        else:

            name = get_object_name(
                object_name,
                count
            )

            position_text = format_positions(
                positions
            )

            if count == 1:
                sentences.append(
                    f"There is a {name} "
                    f"{position_text}."
                )
            else:
                sentences.append(
                    f"There are {count} {name} "
                    f"{position_text}."
                )

    return " ".join(sentences)


def format_positions(positions):
    """
    Convert positions into natural language.
    """

    unique_positions = list(
        dict.fromkeys(positions)
    )

    position_phrases = {
        "left": "on your left",
        "center": "in the center",
        "right": "on your right"
    }

    phrases = [
        position_phrases.get(position, f"at the {position}")
        for position in unique_positions
    ]

    if len(phrases) == 1:
        return phrases[0]

    elif len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"

    else:
        return (
            ", ".join(phrases[:-1])
            + f", and {phrases[-1]}"
        )
def generate_object_location_response(
    object_name,
    locations
):
    """
    Generate a natural-language response
    describing where an object is located.
    """

    if not locations:
        return (
            f"I cannot find a {object_name} "
            "around you."
        )

    position_phrases = {
        "left": "on your left",
        "center": "in the center",
        "right": "on your right"
    }

    phrases = []

    for location in locations:
        phrase = position_phrases.get(
            location,
            f"at the {location}"
        )

        if phrase not in phrases:
            phrases.append(phrase)

    if len(phrases) == 1:
        location_text = phrases[0]

    elif len(phrases) == 2:
        location_text = (
            f"{phrases[0]} and {phrases[1]}"
        )

    else:
        location_text = (
            ", ".join(phrases[:-1])
            + f", and {phrases[-1]}"
        )

    if len(locations) == 1:
        article = "an" if object_name[0] in "aeiou" else "a"

        return (
            f"The {object_name} is "
            f"{location_text}."
        )

    return (
        f"I can see {len(locations)} "
        f"{object_name}s {location_text}."
    )
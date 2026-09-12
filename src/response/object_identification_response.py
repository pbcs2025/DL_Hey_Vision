def generate_object_identification_response(detection):
    """
    Generate a natural-language response for
    the identified object.
    """

    if detection is None:
        return "I cannot identify the object."

    object_name = detection["object"]
    confidence = detection["confidence"]

    if confidence < 0.50:
        return (
            f"I think this may be a {object_name}, "
            f"but I am not very confident."
        )

    return f"This appears to be a {object_name}."
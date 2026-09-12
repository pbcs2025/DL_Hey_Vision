def generate_movement_response(
    movement,
    confidence,
    position=None,
    threshold=0.80
):
    """
    Generate a warning response based on
    movement type, confidence, and position.
    """

    # Ignore uncertain predictions
    if confidence < threshold:
        return None

    # Approaching warning
    if movement == "Approaching":

        if position == "left":
            return "Someone on your left appears to be approaching you."

        if position == "center":
            return "Someone in front of you appears to be approaching you."

        if position == "right":
            return "Someone on your right appears to be approaching you."

        return "Someone appears to be approaching you."


    # Crossing warning
    if movement == "Crossing":

        if position == "left":
            return "Someone on your left appears to be crossing your path."

        if position == "center":
            return "Someone in front of you appears to be crossing your path."

        if position == "right":
            return "Someone on your right appears to be crossing your path."

        return "Someone appears to be crossing your path."


    # No warning for stationary or moving away
    return None
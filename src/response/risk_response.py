def generate_risk_response(
    object_name,
    position,
    risk_level,
    risk_confidence=1.0,
    threshold=0.60
):
    """
    Generate a concise response based on
    predicted risk level and ANN confidence.
    """

    # Do not generate a risk warning when
    # the ANN prediction is too uncertain.
    if risk_confidence < threshold:
        return (
            f"I detect a {object_name} {position}, "
            f"but I am not confident about its risk level."
        )

    if risk_level == "LOW":
        return (
            f"I detect a {object_name} {position}."
        )

    if risk_level == "MEDIUM":
        return (
            f"Caution. I detect a {object_name} "
            f"{position}."
        )

    if risk_level == "HIGH":
        return (
            f"Warning. A {object_name} is "
            f"{position}."
        )

    if risk_level == "CRITICAL":
        return (
            f"Urgent warning. A {object_name} "
            f"is a critical obstacle {position}."
        )

    return (
        f"I detect a {object_name} "
        f"{position}."
    )
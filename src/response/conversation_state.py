# --------------------------------------------------
# Conversation state
# --------------------------------------------------

last_response = None
last_detections = []
last_movement_events = []
last_query = None
last_intent = None


def save_state(
    query,
    intent,
    response,
    detections=None,
    movement_events=None
):
    """
    Store the latest assistant interaction.
    """

    global last_response
    global last_detections
    global last_movement_events
    global last_query
    global last_intent

    last_response = response
    last_detections = detections or []
    last_movement_events = movement_events or []
    last_query = query
    last_intent = intent


def get_last_response():
    return last_response


def get_last_detections():
    return last_detections


def get_last_movement_events():
    return last_movement_events


def get_last_query():
    return last_query


def get_last_intent():
    return last_intent
class AlertManager:
    """
    Manages repeated movement alerts for tracked objects.

    A warning is generated when an object enters
    an alert-worthy movement state:
        Approaching or Crossing

    Repeated warnings for the same track are suppressed
    until the object leaves the warning state.
    """

    def __init__(self):
        self.previous_state = {}

    def should_alert(
        self,
        track_id,
        movement,
        confidence,
        threshold=0.80
    ):
        """
        Decide whether a new movement alert should be generated.
        """

        # Ignore uncertain movement predictions
        if confidence < threshold:
            return False

        previous_movement = self.previous_state.get(
            track_id
        )

        # First observation of this track
        if previous_movement is None:
            return movement in [
                "Approaching",
                "Crossing"
            ]

        was_warning = previous_movement in [
            "Approaching",
            "Crossing"
        ]

        is_warning = movement in [
            "Approaching",
            "Crossing"
        ]

        # Alert only when entering a warning state
        if is_warning and not was_warning:
            return True

        return False

    def update_state(
        self,
        track_id,
        movement
    ):
        """
        Store the latest movement state
        for the tracked object.
        """

        self.previous_state[
            track_id
        ] = movement

    def remove_track(
        self,
        track_id
    ):
        """
        Remove a track when it disappears.
        """

        if track_id in self.previous_state:

            del self.previous_state[
                track_id
            ]
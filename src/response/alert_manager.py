class AlertManager:

    def __init__(self):
        # Stores the last movement state for each tracked object
        self.previous_state = {}

    def should_alert(self, track_id, movement, confidence, threshold=0.80):

        # Ignore uncertain predictions
        if confidence < threshold:
            return False

        previous_movement = self.previous_state.get(track_id)

        # First time seeing this track
        if previous_movement is None:
            return movement in ["Approaching", "Crossing"]

        # Alert only when entering a warning state
        was_warning = previous_movement in ["Approaching", "Crossing"]
        is_warning = movement in ["Approaching", "Crossing"]

        if is_warning and not was_warning:
            return True

        return False

    def update_state(self, track_id, movement):

        self.previous_state[track_id] = movement

    def remove_track(self, track_id):

        if track_id in self.previous_state:
            del self.previous_state[track_id]
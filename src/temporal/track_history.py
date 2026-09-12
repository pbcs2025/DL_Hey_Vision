from collections import defaultdict


class TrackHistory:

    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length

        # Stores feature sequences for each track ID
        self.history = defaultdict(list)

    def update(self, track_id, features):
        """
        Add the latest feature vector for a tracked object.
        """

        self.history[track_id].append(features)

        # Keep only the most recent sequence_length frames
        if len(self.history[track_id]) > self.sequence_length:
            self.history[track_id].pop(0)

    def get_sequence(self, track_id):
        """
        Return the sequence when enough frames are available.
        """

        if len(self.history[track_id]) < self.sequence_length:
            return None

        return self.history[track_id]

    def remove_track(self, track_id):
        """
        Remove a track from history.
        """

        if track_id in self.history:
            del self.history[track_id]
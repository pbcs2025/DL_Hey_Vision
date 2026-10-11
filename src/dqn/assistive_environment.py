import numpy as np


# Actions
WAIT = 0
SPEAK = 1
WARN = 2

ACTION_NAMES = {
    WAIT: "WAIT",
    SPEAK: "SPEAK",
    WARN: "WARN"
}


# Movement encoding used during DQN training
MOVEMENT_VALUES = {
    "stationary": 0.0,
    "moving_away": 0.25,
    "crossing": 0.75,
    "approaching": 1.0
}


class AssistiveEnvironment:

    def __init__(self):
        self.state = None
        self.scenario = None

    def reset(self, scenario):

        self.scenario = scenario

        self.state = np.array([
            scenario["risk"],
            scenario["confidence"],
            MOVEMENT_VALUES[scenario["movement"]],
            scenario["ahead"],
            scenario["priority"],
            scenario["recent_alert"]
        ], dtype=np.float32)

        return self.state

    def calculate_reward(self, action):

        risk = self.scenario["risk"]
        movement = self.scenario["movement"]
        ahead = self.scenario["ahead"]
        priority = self.scenario["priority"]
        recent_alert = self.scenario["recent_alert"]

        # Critical danger directly ahead
        if risk >= 0.75 and ahead == 1:

            # Active approaching/crossing danger
            if movement in ["approaching", "crossing"]:

                if action == WARN:
                    return 10.0

                elif action == SPEAK:
                    return -5.0

                elif action == WAIT:
                    return -10.0

            # Already alerted
            if recent_alert == 1:

                if action == WAIT:
                    return 5.0

                elif action == SPEAK:
                    return -2.0

                elif action == WARN:
                    return -6.0

            if action == WARN:
                return 10.0

            elif action == SPEAK:
                return -5.0

            elif action == WAIT:
                return -10.0

        # Approaching/crossing danger ahead
        if movement in ["approaching", "crossing"] and ahead == 1:

            if action == WARN:
                return 8.0

            elif action == SPEAK:
                return 2.0

            elif action == WAIT:
                return -8.0

        # Important user query
        if priority >= 0.6:

            if action == SPEAK:
                return 5.0

            elif action == WARN:
                return -2.0

            elif action == WAIT:
                return -4.0

        # Recently alerted
        if recent_alert == 1:

            if action == WAIT:
                return 5.0

            elif action == SPEAK:
                return -2.0

            elif action == WARN:
                return -6.0

        # Low-risk situation
        if risk < 0.3:

            if action == WAIT:
                return 4.0

            elif action == SPEAK:
                return 1.0

            elif action == WARN:
                return -5.0

        # Default behavior
        if action == SPEAK:
            return 2.0

        elif action == WAIT:
            return 1.0

        elif action == WARN:
            return -2.0

        return 0.0

    def step(self, action):

        reward = self.calculate_reward(action)

        done = True

        return self.state, reward, done


def get_action_name(action):

    return ACTION_NAMES[action]
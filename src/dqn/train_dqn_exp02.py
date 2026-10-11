
import os
import random
from collections import deque

import numpy as np
import tensorflow as tf

from .dqn_model import build_dqn
from .assistive_environment import AssistiveEnvironment


# --------------------------------------------------
# Configuration
# --------------------------------------------------

STATE_SIZE = 6
ACTION_SIZE = 3

EPISODES = 3000
GAMMA = 0.95
LEARNING_RATE = 0.001

BATCH_SIZE = 64
MEMORY_SIZE = 10000

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

TARGET_UPDATE_FREQUENCY = 100

SEED = 42

MODEL_PATH = "models/dqn/assistive_dqn_exp02.keras"
TARGET_MODEL_PATH = "models/dqn/assistive_dqn_exp02_target.keras"
REWARD_PATH = "experiments/dqn/episode_rewards_exp02.npy"


np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# --------------------------------------------------
# Replay memory
# --------------------------------------------------

class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, action, reward, next_state, done)
        )

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.memory)


# --------------------------------------------------
# Targeted scenario generation
# --------------------------------------------------

def generate_scenario():

    # 60% targeted scenarios
    # 40% general random scenarios

    if random.random() < 0.60:

        scenario_type = random.choice([
            "alert_stationary",
            "alert_moving_away",
            "alert_approaching",
            "alert_crossing",
            "moving_away",
            "general"
        ])

        if scenario_type == "alert_stationary":

            risk = random.uniform(0.3, 1.0)
            confidence = random.uniform(0.8, 1.0)
            movement = "stationary"
            ahead = random.choice([0, 1])
            priority = random.uniform(0.0, 1.0)
            recent_alert = 1

        elif scenario_type == "alert_moving_away":

            risk = random.uniform(0.3, 1.0)
            confidence = random.uniform(0.8, 1.0)
            movement = "moving_away"
            ahead = random.choice([0, 1])
            priority = random.uniform(0.0, 1.0)
            recent_alert = 1

        elif scenario_type == "alert_approaching":

            risk = random.uniform(0.5, 1.0)
            confidence = random.uniform(0.8, 1.0)
            movement = "approaching"
            ahead = 1
            priority = random.uniform(0.0, 1.0)
            recent_alert = 1

        elif scenario_type == "alert_crossing":

            risk = random.uniform(0.5, 1.0)
            confidence = random.uniform(0.8, 1.0)
            movement = "crossing"
            ahead = 1
            priority = random.uniform(0.0, 1.0)
            recent_alert = 1

        elif scenario_type == "moving_away":

            risk = random.uniform(0.0, 1.0)
            confidence = random.uniform(0.6, 1.0)
            movement = "moving_away"
            ahead = random.choice([0, 1])
            priority = random.uniform(0.0, 1.0)
            recent_alert = 0

        else:

            risk = random.uniform(0.0, 1.0)
            confidence = random.uniform(0.6, 1.0)
            movement = random.choice([
                "stationary",
                "moving_away",
                "crossing",
                "approaching"
            ])
            ahead = random.choice([0, 1])
            priority = random.uniform(0.0, 1.0)
            recent_alert = random.choice([0, 1])

    else:

        # General random scenarios for broad coverage

        risk = random.uniform(0.0, 1.0)
        confidence = random.uniform(0.6, 1.0)

        movement = random.choice([
            "stationary",
            "moving_away",
            "crossing",
            "approaching"
        ])

        ahead = random.choice([0, 1])
        priority = random.uniform(0.0, 1.0)
        recent_alert = random.choice([0, 1])

    return {
        "risk": risk,
        "confidence": confidence,
        "movement": movement,
        "ahead": ahead,
        "priority": priority,
        "recent_alert": recent_alert
    }


# --------------------------------------------------
# DQN training step
# --------------------------------------------------

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)


@tf.function
def train_step(
    online_network,
    target_network,
    states,
    actions,
    rewards,
    next_states,
    dones
):

    with tf.GradientTape() as tape:

        current_q = online_network(states, training=True)

        next_q = target_network(
            next_states,
            training=False
        )

        max_next_q = tf.reduce_max(
            next_q,
            axis=1
        )

        targets = rewards + (
            GAMMA *
            max_next_q *
            (1.0 - dones)
        )

        action_mask = tf.one_hot(
            actions,
            ACTION_SIZE
        )

        selected_q = tf.reduce_sum(
            current_q * action_mask,
            axis=1
        )

        loss = tf.reduce_mean(
            tf.square(targets - selected_q)
        )

    gradients = tape.gradient(
        loss,
        online_network.trainable_variables
    )

    optimizer.apply_gradients(
        zip(
            gradients,
            online_network.trainable_variables
        )
    )

    return loss


# --------------------------------------------------
# Initialize networks
# --------------------------------------------------

online_network = build_dqn()
target_network = build_dqn()

target_network.set_weights(
    online_network.get_weights()
)

memory = ReplayMemory(MEMORY_SIZE)

environment = AssistiveEnvironment()

epsilon = EPSILON_START

episode_rewards = []


# --------------------------------------------------
# Training
# --------------------------------------------------

for episode in range(1, EPISODES + 1):

    scenario = generate_scenario()

    state = environment.reset(scenario)

    # Epsilon-greedy action

    if random.random() < epsilon:

        action = random.randrange(ACTION_SIZE)

    else:

        q_values = online_network.predict(
            np.expand_dims(state, axis=0),
            verbose=0
        )[0]

        action = int(np.argmax(q_values))


    next_state, reward, done = environment.step(action)

    memory.add(
        state,
        action,
        reward,
        next_state,
        done
    )

    episode_rewards.append(reward)


    # Replay training

    if len(memory) >= BATCH_SIZE:

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = memory.sample(BATCH_SIZE)

        train_step(
            online_network,
            target_network,
            states,
            actions,
            rewards,
            next_states,
            dones
        )


    # Target network update

    if episode % TARGET_UPDATE_FREQUENCY == 0:

        target_network.set_weights(
            online_network.get_weights()
        )


    # Epsilon decay

    epsilon = max(
        EPSILON_MIN,
        epsilon * EPSILON_DECAY
    )


    # Progress

    if episode % 100 == 0:

        recent_rewards = episode_rewards[-100:]

        average_reward = np.mean(
            recent_rewards
        )

        print(
            f"Episode {episode:4d} | "
            f"Average reward: "
            f"{average_reward:7.3f} | "
            f"Epsilon: {epsilon:.3f}"
        )


# --------------------------------------------------
# Save models and rewards
# --------------------------------------------------

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)

os.makedirs(
    os.path.dirname(REWARD_PATH),
    exist_ok=True
)

online_network.save(MODEL_PATH)
target_network.save(TARGET_MODEL_PATH)

np.save(
    REWARD_PATH,
    np.array(episode_rewards)
)

print("\nDQN Experiment 02 training completed.")
print("Model saved to:", MODEL_PATH)
print("Target model saved to:", TARGET_MODEL_PATH)
print("Rewards saved to:", REWARD_PATH)

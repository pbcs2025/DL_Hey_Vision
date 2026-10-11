
import os
import random
from collections import deque

import numpy as np
import tensorflow as tf

from .dqn_model import build_dqn
from .assistive_environment import (
    AssistiveEnvironment,
    WAIT,
    SPEAK,
    WARN
)


# -----------------------------
# Configuration
# -----------------------------

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


# -----------------------------
# Reproducibility
# -----------------------------

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# -----------------------------
# Paths
# -----------------------------

MODEL_DIR = "/kaggle/working/AI_Vision_Assistant/models/dqn"
EXPERIMENT_DIR = "/kaggle/working/AI_Vision_Assistant/experiments/dqn"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EXPERIMENT_DIR, exist_ok=True)


# -----------------------------
# Scenario generation
# -----------------------------

def generate_scenario():

    risk = np.random.uniform(0.0, 1.0)

    confidence = np.random.uniform(0.6, 1.0)

    movement = random.choice([
        "stationary",
        "moving_away",
        "crossing",
        "approaching"
    ])

    ahead = random.choice([0, 1])

    priority = np.random.uniform(0.0, 1.0)

    recent_alert = random.choice([0, 1])

    return {
        "risk": risk,
        "confidence": confidence,
        "movement": movement,
        "ahead": ahead,
        "priority": priority,
        "recent_alert": recent_alert
    }


# -----------------------------
# Replay memory
# -----------------------------

class ReplayMemory:

    def __init__(self, max_size):

        self.memory = deque(
            maxlen=max_size
        )

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )

    def sample(self, batch_size):

        batch = random.sample(
            self.memory,
            batch_size
        )

        states = np.array(
            [item[0] for item in batch],
            dtype=np.float32
        )

        actions = np.array(
            [item[1] for item in batch],
            dtype=np.int32
        )

        rewards = np.array(
            [item[2] for item in batch],
            dtype=np.float32
        )

        next_states = np.array(
            [item[3] for item in batch],
            dtype=np.float32
        )

        dones = np.array(
            [item[4] for item in batch],
            dtype=np.float32
        )

        return (
            states,
            actions,
            rewards,
            next_states,
            dones
        )

    def __len__(self):

        return len(self.memory)


# -----------------------------
# Training
# -----------------------------

env = AssistiveEnvironment()

online_network = build_dqn()

target_network = build_dqn()

target_network.set_weights(
    online_network.get_weights()
)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

loss_function = tf.keras.losses.MeanSquaredError()

memory = ReplayMemory(MEMORY_SIZE)

epsilon = EPSILON_START

episode_rewards = []


def train_step(
    states,
    actions,
    rewards,
    next_states,
    dones
):

    with tf.GradientTape() as tape:

        current_q_values = online_network(
            states,
            training=True
        )

        next_q_values = target_network(
            next_states,
            training=False
        )

        max_next_q_values = tf.reduce_max(
            next_q_values,
            axis=1
        )

        target_q_values = (
            rewards
            + GAMMA
            * max_next_q_values
            * (1.0 - dones)
        )

        action_mask = tf.one_hot(
            actions,
            ACTION_SIZE
        )

        selected_q_values = tf.reduce_sum(
            current_q_values * action_mask,
            axis=1
        )

        loss = loss_function(
            target_q_values,
            selected_q_values
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


# -----------------------------
# Main training loop
# -----------------------------

for episode in range(1, EPISODES + 1):

    scenario = generate_scenario()

    state = env.reset(
        scenario
    )

    # Epsilon-greedy action selection

    if np.random.random() < epsilon:

        action = np.random.randint(
            ACTION_SIZE
        )

    else:

        q_values = online_network(
            np.expand_dims(
                state,
                axis=0
            ),
            training=False
        ).numpy()[0]

        action = int(
            np.argmax(q_values)
        )

    next_state, reward, done = env.step(
        action
    )

    memory.add(
        state,
        action,
        reward,
        next_state,
        done
    )

    # Train only after enough experiences exist

    if len(memory) >= BATCH_SIZE:

        batch = memory.sample(
            BATCH_SIZE
        )

        loss = train_step(
            *batch
        )

    else:

        loss = None

    episode_rewards.append(
        reward
    )

    # Decay exploration

    epsilon = max(
        EPSILON_MIN,
        epsilon * EPSILON_DECAY
    )

    # Update target network

    if episode % TARGET_UPDATE_FREQUENCY == 0:

        target_network.set_weights(
            online_network.get_weights()
        )

    # Progress output

    if episode % 100 == 0:

        recent_rewards = episode_rewards[-100:]

        average_reward = np.mean(
            recent_rewards
        )

        print(
            f"Episode {episode:4d} | "
            f"Average reward: {average_reward:7.3f} | "
            f"Epsilon: {epsilon:.3f}"
        )


# -----------------------------
# Save model
# -----------------------------

online_network.save(
    os.path.join(
        MODEL_DIR,
        "assistive_dqn.keras"
    )
)

target_network.save(
    os.path.join(
        MODEL_DIR,
        "assistive_dqn_target.keras"
    )
)


# Save reward history

np.save(
    os.path.join(
        EXPERIMENT_DIR,
        "episode_rewards.npy"
    ),
    np.array(
        episode_rewards
    )
)

print()
print("DQN training completed.")
print(
    "Model saved to:",
    os.path.join(
        MODEL_DIR,
        "assistive_dqn.keras"
    )
)

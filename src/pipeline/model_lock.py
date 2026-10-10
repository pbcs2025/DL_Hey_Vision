"""
Central inference synchronization lock for Deep Learning models.
Prevents concurrent multithreaded evaluation across YOLO, Whisper,
Keras Intent NLP, Approaching LSTM, Risk ANN, and Assistive DQN.
"""

import threading

# Global recursive lock for thread-safe model inference
MODEL_LOCK = threading.RLock()


def acquire_model_lock():
    """Acquires the shared model inference lock."""
    MODEL_LOCK.acquire()


def release_model_lock():
    """Releases the shared model inference lock."""
    MODEL_LOCK.release()

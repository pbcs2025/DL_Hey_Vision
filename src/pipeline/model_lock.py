"""
Central inference synchronization locks for Deep Learning models.
Provides granular locks per model family so that heavy speech transcription (Whisper)
does not stall real-time video object detection (YOLO) or decision reasoning (Keras).
"""

import threading

# Dedicated lock for YOLO11n object detection
YOLO_LOCK = threading.RLock()

# Dedicated lock for OpenAI Whisper speech-to-text
WHISPER_LOCK = threading.RLock()

# Dedicated lock for TensorFlow / Keras models (Intent NLP, Approaching LSTM, Risk ANN, Assistive DQN)
KERAS_LOCK = threading.RLock()

# Unified fallback lock for full pipeline synchronization if needed
MODEL_LOCK = threading.RLock()


"""
Central inference synchronization locks for Deep Learning models.
Provides granular locks per model family so that heavy speech transcription (Whisper)
does not stall real-time video object detection (YOLO) or decision reasoning (Keras).
"""

import os
import threading

DEFAULT_THREADS = max(1, (os.cpu_count() or 4) // 2)
NUM_THREADS = int(os.environ.get("VISION_NUM_THREADS", DEFAULT_THREADS))
INTER_OP_THREADS = max(1, NUM_THREADS // 2)

os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(NUM_THREADS)
os.environ["TF_NUM_INTEROP_THREADS"] = str(INTER_OP_THREADS)

try:
    import torch
    torch.set_num_threads(NUM_THREADS)
except Exception:
    pass

try:
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(INTER_OP_THREADS)
except Exception:
    pass

# Dedicated lock for YOLO11n object detection
YOLO_LOCK = threading.RLock()

# Dedicated lock for OpenAI Whisper speech-to-text
WHISPER_LOCK = threading.RLock()

# Dedicated lock for TensorFlow / Keras models (Intent NLP, Approaching LSTM, Risk ANN, Assistive DQN)
KERAS_LOCK = threading.RLock()

# Unified fallback lock for full pipeline synchronization if needed
MODEL_LOCK = threading.RLock()


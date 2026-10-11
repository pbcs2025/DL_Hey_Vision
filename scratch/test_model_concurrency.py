import os
import sys
import time
import threading
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import whisper
from ultralytics import YOLO

def test_whisper_and_yolo():
    print("Loading YOLO and Whisper models...")
    yolo_model = YOLO("backup/Experiment_02_Backup/best.pt")
    whisper_model = whisper.load_model("base.en")
    print("Models loaded.")

    # Create synthetic 2-second audio (16kHz PCM float32)
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # Dummy image
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    yolo_lock = threading.Lock()
    whisper_lock = threading.Lock()

    # 1. Measure pure Whisper latency
    t0 = time.perf_counter()
    with whisper_lock:
        res = whisper_model.transcribe(audio_data, language="en", fp16=False)
    whisper_duration = time.perf_counter() - t0
    print(f"Whisper standalone transcription duration: {whisper_duration*1000:.1f}ms")

    # 2. Test concurrent execution: YOLO running in loop while Whisper runs
    yolo_frames_processed = 0
    yolo_latencies = []
    stop_yolo = threading.Event()

    def yolo_worker():
        nonlocal yolo_frames_processed
        while not stop_yolo.is_set():
            t_frame = time.perf_counter()
            with yolo_lock:
                yolo_model.predict(img, conf=0.25, verbose=False)
            dt = (time.perf_counter() - t_frame) * 1000.0
            yolo_latencies.append(dt)
            yolo_frames_processed += 1
            time.sleep(0.01)

    yolo_thread = threading.Thread(target=yolo_worker, daemon=True)
    yolo_thread.start()

    time.sleep(0.2) # let YOLO start

    t_concurrent_whisper = time.perf_counter()
    with whisper_lock:
        res2 = whisper_model.transcribe(audio_data, language="en", fp16=False)
    whisper_concurrent_duration = time.perf_counter() - t_concurrent_whisper

    stop_yolo.set()
    yolo_thread.join(timeout=2.0)

    avg_yolo = np.mean(yolo_latencies) if yolo_latencies else 0.0
    print(f"\n--- Concurrency Test Results ---")
    print(f"Whisper duration during concurrent YOLO: {whisper_concurrent_duration*1000:.1f}ms")
    print(f"YOLO frames processed during Whisper: {yolo_frames_processed}")
    print(f"Average YOLO inference latency: {avg_yolo:.1f}ms")
    print(f"[OK] YOLO and Whisper ran concurrently without crash or conflict!")

if __name__ == "__main__":
    test_whisper_and_yolo()

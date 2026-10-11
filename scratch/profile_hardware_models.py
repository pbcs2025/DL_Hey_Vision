import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import torch
import numpy as np
from ultralytics import YOLO
import whisper

def check_hardware_and_benchmark():
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    print(f"CUDA Available: {cuda_avail}")
    print(f"Device Name: {device_name}")

    # Generate synthetic 2s speech audio (16kHz PCM float32)
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    synthetic_audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # 1. Benchmark Whisper tiny.en vs base.en
    for model_name in ["tiny.en", "base.en"]:
        print(f"\nLoading Whisper {model_name}...")
        w_model = whisper.load_model(model_name)
        # Warmup
        w_model.transcribe(synthetic_audio, language="en", fp16=False)
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            w_model.transcribe(synthetic_audio, language="en", fp16=False)
            times.append((time.perf_counter() - t0) * 1000.0)
        print(f"Whisper {model_name} Mean Transcription Latency: {np.mean(times):.1f} ms (p50: {np.median(times):.1f} ms)")

    # 2. Benchmark YOLO imgsz 416 vs 640
    print("\nLoading YOLO model...")
    y_model = YOLO("backup/Experiment_02_Backup/best.pt")
    sample_img_path = "data/test/door_test.png"

    for size in [640, 416]:
        # Warmup
        y_model.predict(sample_img_path, imgsz=size, verbose=False)
        times = []
        res = None
        for _ in range(15):
            t0 = time.perf_counter()
            res = y_model.predict(sample_img_path, imgsz=size, verbose=False)
            times.append((time.perf_counter() - t0) * 1000.0)
        mean_ms = np.mean(times)
        fps = 1000.0 / mean_ms
        boxes = len(res[0].boxes) if res and res[0].boxes is not None else 0
        top_conf = float(res[0].boxes.conf[0]) if boxes > 0 else 0.0
        print(f"YOLO imgsz={size}: Mean Latency = {mean_ms:.1f} ms | FPS = {fps:.1f} | Detections = {boxes} (top conf: {top_conf:.2f})")

if __name__ == "__main__":
    check_hardware_and_benchmark()

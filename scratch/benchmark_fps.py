import time
import numpy as np
from ultralytics import YOLO

def benchmark():
    model = YOLO("backup/Experiment_02_Backup/best.pt")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Warmup
    for _ in range(5):
        model.predict(frame, verbose=False)

    latencies = []
    for _ in range(30):
        t0 = time.perf_counter()
        model.predict(frame, verbose=False)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

    mean_ms = np.mean(latencies)
    min_ms = np.min(latencies)
    max_ms = np.max(latencies)
    p50_ms = np.median(latencies)
    det_fps = 1000.0 / mean_ms

    print("--- Real Measured Performance on this Machine ---")
    print(f"Hardware: CPU (native Windows)")
    print(f"Input Resolution: 640x480")
    print(f"Mean Inference Latency: {mean_ms:.1f} ms")
    print(f"Median (p50) Latency:   {p50_ms:.1f} ms")
    print(f"Min / Max Latency:      {min_ms:.1f} ms / {max_ms:.1f} ms")
    print(f"Detection Throughput:   {det_fps:.1f} FPS")

if __name__ == "__main__":
    benchmark()

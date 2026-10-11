"""
Benchmark movement analysis speed on 10s video clip.
Tests:
- Quick Analysis: stride=3, imgsz=416, direct callable Keras.
- Measures time per frame and total elapsed time.
"""
import time
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.video_assistant_pipeline import process_video

clip_path = os.path.join(PROJECT_ROOT, "scratch", "test_clip_10s.mp4")
if not os.path.exists(clip_path):
    print("test_clip_10s.mp4 not found, creating from test_video.mp4...")
    import cv2
    cap = cv2.VideoCapture(os.path.join(PROJECT_ROOT, "data", "test", "test_video.mp4"))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(clip_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    count = 0
    while count < 301:
        ret, frame = cap.read()
        if not ret: break
        out.write(frame)
        count += 1
    cap.release()
    out.release()
    print("Created test_clip_10s.mp4 with", count, "frames")

print("=" * 60)
print("BENCHMARKING QUICK MOVEMENT ANALYSIS (stride=3, imgsz=416)")
print("=" * 60)

t0 = time.perf_counter()
events = process_video(clip_path, stride=3, imgsz=416)
total_time = time.perf_counter() - t0

total_frames = 301
time_per_frame = (total_time / total_frames) * 1000.0  # ms per input video frame
processed_frames = (total_frames + 2) // 3
time_per_processed_frame = (total_time / processed_frames) * 1000.0

print("\n" + "=" * 60)
print("BENCHMARK RESULTS:")
print(f"Total Video Duration: 10.0 s (301 frames)")
print(f"Total Processing Time: {total_time:.2f} s")
print(f"Time Per Video Frame: {time_per_frame:.1f} ms")
print(f"Time Per Inferred Frame (stride=3): {time_per_processed_frame:.1f} ms")
print(f"Events Triggered: {len(events)}")
print("=" * 60)

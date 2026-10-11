"""
Script to generate precomputed, cached movement analysis results for sample videos.
Stores results in data/cache/<video_basename>_movement_cache.json.
Format:
{
  "source_video": "test_video.mp4",
  "label": "pre-analysed run (2026-10-11, 15.0s / 453 frames)",
  "date": "2026-10-11",
  "duration_sec": 15.0,
  "total_frames": 453,
  "events_count": <int>,
  "generated_at": "<ISO timestamp>",
  "stride": 1,
  "imgsz": 640,
  "events": [...]
}
"""

import os
import sys
import time
import json
import datetime
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.video_assistant_pipeline import process_video

CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "cache")

def generate_cache_for_video(video_path: str, stride: int = 1, imgsz: int = 640):
    if not os.path.exists(video_path):
        print(f"[GenerateCache] Video file not found: {video_path}")
        return None

    os.makedirs(CACHE_DIR, exist_ok=True)
    basename = os.path.basename(video_path)
    stem = os.path.splitext(basename)[0]
    out_file = os.path.join(CACHE_DIR, f"{stem}_movement_cache.json")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    duration_sec = round(total_frames / max(1.0, fps), 1)
    today_str = datetime.date.today().isoformat()
    label = f"pre-analysed run ({today_str}, {duration_sec}s / {total_frames} frames)"

    print(f"\n[GenerateCache] Processing {basename} ({total_frames} frames, {duration_sec}s)...")
    t0 = time.perf_counter()
    events = process_video(video_path, stride=stride, imgsz=imgsz)
    elapsed = round(time.perf_counter() - t0, 2)

    cache_data = {
        "source_video": basename,
        "label": label,
        "date": today_str,
        "duration_sec": duration_sec,
        "total_frames": total_frames,
        "events_count": len(events),
        "generated_at": datetime.datetime.now().isoformat(),
        "elapsed_sec": elapsed,
        "stride": stride,
        "imgsz": imgsz,
        "events": events
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2)

    print(f"[GenerateCache] Saved cache to: {out_file} (events={len(events)}, elapsed={elapsed}s)")
    return cache_data

def main():
    sample_videos = [
        os.path.join(PROJECT_ROOT, "data", "test", "test_video.mp4"),
        os.path.join(PROJECT_ROOT, "scratch", "test_clip_10s.mp4")
    ]
    for v in sample_videos:
        if os.path.exists(v):
            generate_cache_for_video(v, stride=1, imgsz=640)

if __name__ == "__main__":
    main()

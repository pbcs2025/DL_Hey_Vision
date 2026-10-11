import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ui.backend.video_streamer import CameraStreamer, AsyncDetectionWorker
from ui.backend.hazard_monitor import ProactiveHazardMonitor

cam = CameraStreamer()
cam.set_source("video", os.path.join(PROJECT_ROOT, "data", "test", "test_video.mp4"))
cam.start()

det = AsyncDetectionWorker(cam, conf_threshold=0.25, imgsz=416)
det.start()
time.sleep(1.0)

def get_latest_det():
    _, dets, det_fps, _, frame_id, det_ts = det.get_latest_data()
    return dets, det_fps, frame_id, det_ts

m = ProactiveHazardMonitor(get_latest_det, lambda: "STANDBY")
m.start()
time.sleep(30.0)
m.stop()
det.stop()
cam.release()

log = m.get_decision_log()
spoken = [e for e in log if e.get("spoke_alert")]
print(f"Total spoken alerts in 30s: {len(spoken)}")
for e in spoken:
    print(f"{e['time_str']} | {e['object']} ({e['position']}, {e['proximity']}) -> {e['risk']} ({e['risk_confidence']}) | {e['raw_action']} | phrase: {e['phrase']}")

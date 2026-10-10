import os
import sys
import time
import unittest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ui.backend.app_server import app, generate_mjpeg_stream
from ui.backend.video_streamer import CameraStreamer, AsyncDetectionWorker


class TestPhase2Backend(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_camera_streamer_lifecycle(self):
        """Verify CameraStreamer starts, acquires frames or fallback frame, and releases cleanly."""
        streamer = CameraStreamer(preferred_index=0)
        streamer.start()
        time.sleep(0.3)

        frame = streamer.get_raw_frame()
        self.assertIsNotNone(frame, "Streamer must provide a frame (camera or offline fallback).")
        self.assertEqual(len(frame.shape), 3, "Frame must have 3 color channels.")
        self.assertGreaterEqual(streamer.fps, 0.0)

        # Clean release
        streamer.release()
        self.assertFalse(streamer.running)

    def test_async_detection_worker_lifecycle(self):
        """Verify AsyncDetectionWorker runs inference, sets real metrics, and never says approaching."""
        streamer = CameraStreamer(preferred_index=0)
        streamer.start()
        worker = AsyncDetectionWorker(streamer, conf_threshold=0.25)
        worker.start()

        # Allow worker to process a few frames
        time.sleep(1.2)

        annotated, detections, det_fps, det_lat_ms = worker.get_latest_data()
        self.assertIsNotNone(annotated, "Worker must produce an annotated frame.")
        self.assertIsInstance(detections, list)
        self.assertGreaterEqual(det_lat_ms, 0.0, "Detection latency must be non-negative.")
        self.assertGreaterEqual(det_fps, 0.0, "Detection FPS must be non-negative.")

        # Ensure no detection dictionary uses the forbidden word 'approaching'
        for d in detections:
            proximity = d.get("proximity", "").lower()
            self.assertNotEqual(proximity, "approaching", "Walk Mode must NEVER use 'approaching'.")
            self.assertIn(proximity, ["close", "nearby", "far"])

        worker.stop()
        streamer.release()

    def test_api_health_endpoint(self):
        """Verify /api/health returns 4-point readiness checklist."""
        with TestClient(app) as client:
            resp = client.get("/api/health")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("checklist", data)
            checklist = data["checklist"]

            # 1. Models
            self.assertIn("models", checklist)
            self.assertEqual(checklist["models"]["yolo_classes_count"], 13)
            self.assertIn("door", checklist["models"]["classes"])

            # 2. Camera
            self.assertIn("camera", checklist)
            self.assertIn("available", checklist["camera"])

            # 3. Mic
            self.assertIn("microphone", checklist)
            self.assertIn("available", checklist["microphone"])

            # 4. TTS
            self.assertIn("tts", checklist)
            self.assertTrue(checklist["tts"]["available"])

    def test_api_telemetry_endpoint(self):
        """Verify /api/telemetry returns real measured numbers."""
        with TestClient(app) as client:
            resp = client.get("/api/telemetry")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("camera_fps", data)
            self.assertIn("detection_fps", data)
            self.assertIn("detection_latency_ms", data)
            self.assertIn("detections", data)

    def test_video_feed_mjpeg_stream(self):
        """Verify MJPEG streaming generator yields valid multipart JPEG boundary frames."""
        # 1. Verify route registration
        routes = [route.path for route in app.routes]
        self.assertIn("/video_feed", routes)

        # 2. Verify generator yields valid multipart frame
        generator = generate_mjpeg_stream()
        frame_chunk = next(generator)
        self.assertIsInstance(frame_chunk, bytes)
        self.assertTrue(frame_chunk.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"))
        self.assertGreater(len(frame_chunk), 500)


if __name__ == "__main__":
    unittest.main()

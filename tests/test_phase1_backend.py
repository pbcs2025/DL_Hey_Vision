import os
import sys
import time
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.voice.tts_worker import TTSWorker, get_tts_worker
from src.pipeline.model_lock import MODEL_LOCK
from src.pipeline.assistant_pipeline import (
    process_query,
    get_yolo_classes,
    detect_objects
)


class TestPhase1Backend(unittest.TestCase):

    def test_tts_worker_priority_and_interrupt(self):
        """Verify TTS worker can enqueue normal speech and execute priority interrupt cleanly."""
        worker = get_tts_worker()
        self.assertIsNotNone(worker)

        # Enqueue long lower-priority speech
        worker.speak("This is a long introductory sentence for assistive testing.", priority=10)
        time.sleep(0.1)

        # Verify priority warning interrupt
        worker.interrupt_and_warn("Warning: Hazard ahead!")
        time.sleep(0.5)

        # Worker should remain responsive
        self.assertIsInstance(worker.is_speaking(), bool)

    def test_dynamic_yolo_classes(self):
        """Verify 13 classes are dynamically retrieved directly from yolo_model.names."""
        classes = get_yolo_classes()
        self.assertIsInstance(classes, dict)
        self.assertEqual(len(classes), 13, f"Expected 13 classes from trained YOLO model, got {len(classes)}")
        self.assertIn("door", classes.values(), "Trained model must include the custom 'door' class.")
        self.assertIn("person", classes.values(), "Trained model must include 'person' class.")

    def test_model_lock_concurrency(self):
        """Verify MODEL_LOCK is re-entrant and functional."""
        with MODEL_LOCK:
            self.assertTrue(MODEL_LOCK._is_owned())

    def test_process_query_telemetry_and_latencies(self):
        """Verify process_query returns real measured stage latencies, risk, and DQN action."""
        sample_img = os.path.join(PROJECT_ROOT, "data", "test", "door_test.png")
        if not os.path.exists(sample_img):
            sample_img = os.path.join(PROJECT_ROOT, "data", "test", "test_scene.jpg")

        self.assertTrue(os.path.exists(sample_img), f"Sample test image not found at {sample_img}")

        result = process_query("Where is the door?", sample_img)

        # Verify existing contract is maintained
        self.assertIn("query", result)
        self.assertIn("intent", result)
        self.assertIn("intent_confidence", result)
        self.assertIn("detections", result)
        self.assertIn("response", result)

        # Verify additive telemetry contract
        self.assertIn("risk", result)
        self.assertIn(result["risk"], ["LOW", "MEDIUM", "HIGH", "CRITICAL"])

        self.assertIn("dqn_action", result)
        self.assertIn(result["dqn_action"], ["WAIT", "SPEAK", "WARN"])

        self.assertIn("q_values", result)
        self.assertEqual(len(result["q_values"]), 3)

        self.assertIn("stage_latencies", result)
        latencies = result["stage_latencies"]
        for stage in ["camera_ms", "yolo_ms", "nlp_ms", "spatial_ms", "risk_ms", "dqn_ms", "total_ms"]:
            self.assertIn(stage, latencies, f"Missing stage latency: {stage}")
            self.assertGreaterEqual(latencies[stage], 0.0, f"Latency for {stage} must be non-negative")

        self.assertIn("latency_ms", result)
        self.assertGreater(result["latency_ms"], 0.0, "Total measured latency must be greater than zero")


if __name__ == "__main__":
    unittest.main()

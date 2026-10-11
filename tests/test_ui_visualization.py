import os
import sys
import unittest
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.vision.detection_visualizer import draw_detections, get_scene_summary_data, get_class_color
from src.vision.live_camera import WebcamStream
from src.pipeline.assistant_pipeline import detect_objects, process_query


class TestUIVisualization(unittest.TestCase):

    def setUp(self):
        self.sample_image_path = os.path.join(PROJECT_ROOT, "data", "test", "test_scene.jpg")

    def test_class_color_consistency(self):
        """Ensure the same object class receives the exact same color throughout."""
        color1 = get_class_color("person")
        color2 = get_class_color("person")
        self.assertEqual(color1, color2)

        color_chair1 = get_class_color("chair")
        color_chair2 = get_class_color("chair")
        self.assertEqual(color_chair1, color_chair2)
        self.assertNotEqual(color1, color_chair1)

    def test_draw_detections_on_image(self):
        """Test drawing YOLO bounding boxes and labels on an image."""
        if not os.path.exists(self.sample_image_path):
            self.skipTest("Sample image not found")

        img = cv2.imread(self.sample_image_path)
        self.assertIsNotNone(img)

        detections = detect_objects(img)
        self.assertIsInstance(detections, list)

        annotated = draw_detections(img, detections)
        self.assertIsNotNone(annotated)
        self.assertEqual(annotated.shape, img.shape)

    def test_scene_summary_data(self):
        """Test scene summary data formatting."""
        mock_detections = [
            {"object": "person", "confidence": 0.92},
            {"object": "person", "confidence": 0.88},
            {"object": "chair", "confidence": 0.78},
            {"object": "table", "confidence": 0.85},
        ]
        summary = get_scene_summary_data(mock_detections)
        self.assertEqual(len(summary), 3)

        person_item = next(item for item in summary if item["object"] == "Person")
        self.assertEqual(person_item["count"], 2)
        self.assertEqual(person_item["icon"], "👤")

    def test_webcam_stream_fallback(self):
        """Test that WebcamStream initializes without throwing unhandled exceptions."""
        stream = WebcamStream(camera_index=99)  # Non-existent index
        self.assertFalse(stream.is_available())
        stream.stop()

    def test_pipeline_with_bounding_boxes(self):
        """Test that process_query returns detections with bbox coordinates."""
        if not os.path.exists(self.sample_image_path):
            self.skipTest("Sample image not found")

        result = process_query("What is around me?", self.sample_image_path)
        self.assertIn("response", result)
        self.assertIn("detections", result)
        
        for det in result["detections"]:
            self.assertIn("object", det)
            self.assertIn("confidence", det)
            self.assertIn("bbox", det)
            self.assertEqual(len(det["bbox"]), 4)


if __name__ == "__main__":
    unittest.main()

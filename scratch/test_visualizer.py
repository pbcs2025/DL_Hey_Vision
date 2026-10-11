import os
import sys
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.vision.detection_visualizer import draw_detections, get_scene_summary_data, get_class_color
from src.vision.live_camera import WebcamStream
from src.pipeline.assistant_pipeline import detect_objects

def test_visualizer():
    print("Testing DetectionVisualizer...")

    # Test colors
    color1 = get_class_color("person")
    color2 = get_class_color("person")
    assert color1 == color2, "Color must be consistent for same class"
    print(f"Color for 'person': {color1}")

    # Test sample image
    sample_image_path = os.path.join(PROJECT_ROOT, "data", "test", "test_scene.jpg")
    if os.path.exists(sample_image_path):
        img = cv2.imread(sample_image_path)
        detections = detect_objects(img)
        print(f"Detections count: {len(detections)}")
        for d in detections:
            print(f"  Object: {d['object']}, Conf: {d['confidence']:.2f}, Pos: {d['position']}, BBox: {d.get('bbox')}")
        
        annotated = draw_detections(img, detections)
        summary = get_scene_summary_data(detections)
        print(f"Scene summary items: {summary}")
        print("Visualizer test on sample image PASSED.")

    # Test WebcamStream
    print("\nTesting WebcamStream...")
    stream = WebcamStream(camera_index=0)
    time_spent = 0
    import time
    time.sleep(0.5)
    if stream.is_available():
        frame = stream.get_frame()
        assert frame is not None, "Frame should not be None when camera is available"
        print(f"Webcam frame acquired successfully with shape: {frame.shape}")
    else:
        print("Webcam not available on this machine/environment (graceful fallback working).")
    stream.stop()
    print("ALL TESTS PASSED.")

if __name__ == "__main__":
    test_visualizer()

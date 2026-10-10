"""
Decoupled Video Streamer and Asynchronous Object Detection Engine.
Streams video at camera rate while running YOLO detection asynchronously on the latest frame.
Handles missing/busy cameras gracefully and shuts down cleanly.
"""

import time
import threading
from typing import Optional, List, Dict, Any, Tuple
import cv2
import numpy as np

from src.pipeline.model_lock import YOLO_LOCK
from src.pipeline.assistant_pipeline import yolo_model, get_yolo_classes
from src.vision.detection_visualizer import get_class_color


class CameraStreamer:
    """
    Manages continuous video frame capture from a webcam or video source.
    Decoupled from model inference so the camera feed never stalls.
    """

    def __init__(self, preferred_index: int = 0):
        self.preferred_index = preferred_index
        self.active_index = -1
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_available = False
        self.status_message = "Initializing camera..."
        self.lock = threading.Lock()
        self.latest_raw_frame: Optional[np.ndarray] = None
        self.running = False
        self.fps = 0.0
        self._frame_count = 0
        self._last_fps_calc = time.perf_counter()
        self._thread: Optional[threading.Thread] = None

        self._initialize_camera()

    def _initialize_camera(self):
        """Attempts to open preferred camera index, then falls back to index 1."""
        indices_to_try = [self.preferred_index]
        if self.preferred_index != 1:
            indices_to_try.append(1)
        if self.preferred_index != 0 and 0 not in indices_to_try:
            indices_to_try.append(0)

        for idx in indices_to_try:
            try:
                # Try directshow backend first on Windows for faster initialization, then default
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(idx)

                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        self.cap = cap
                        self.active_index = idx
                        self.is_available = True
                        self.latest_raw_frame = test_frame
                        self.status_message = f"Webcam active on index {idx}."
                        print(f"[CameraStreamer] {self.status_message}")
                        return
                    else:
                        cap.release()
            except Exception as e:
                print(f"[CameraStreamer] Probe failed for index {idx}: {e}")

        self.is_available = False
        self.status_message = "No physical camera detected (indices 0, 1 probed)."
        print(f"[CameraStreamer] {self.status_message}")

    def start(self):
        """Starts the background camera acquisition loop."""
        self.running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="CameraCaptureThread",
            daemon=True
        )
        self._thread.start()

    def _capture_loop(self):
        """Continuous frame grab loop at hardware frame rate."""
        while self.running:
            if self.is_available and self.cap is not None:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.latest_raw_frame = frame
                    self._update_fps()
                else:
                    # Brief pause if frame read failed
                    time.sleep(0.02)
            else:
                # Standby synthetic frame when camera is not detected
                with self.lock:
                    self.latest_raw_frame = self._generate_offline_frame()
                self._update_fps()
                time.sleep(0.06)

    def _update_fps(self):
        self._frame_count += 1
        now = time.perf_counter()
        dt = now - self._last_fps_calc
        if dt >= 1.0:
            instant_fps = self._frame_count / dt
            self.fps = round(0.7 * self.fps + 0.3 * instant_fps, 1) if self.fps > 0 else round(instant_fps, 1)
            self._frame_count = 0
            self._last_fps_calc = now

    def _generate_offline_frame(self) -> np.ndarray:
        """Generates an aesthetic obsidian standby frame when no physical webcam is attached."""
        h, w = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Deep obsidian gradient background
        frame[:] = (20, 14, 10) # BGR obsidian

        # Subtle HUD Reticle brackets
        margin = 40
        bracket_len = 35
        color_cyan = (255, 240, 0)
        # Top-left
        cv2.line(frame, (margin, margin), (margin + bracket_len, margin), color_cyan, 2)
        cv2.line(frame, (margin, margin), (margin, margin + bracket_len), color_cyan, 2)
        # Top-right
        cv2.line(frame, (w - margin, margin), (w - margin - bracket_len, margin), color_cyan, 2)
        cv2.line(frame, (w - margin, margin), (w - margin, margin + bracket_len), color_cyan, 2)
        # Bottom-left
        cv2.line(frame, (margin, h - margin), (margin + bracket_len, h - margin), color_cyan, 2)
        cv2.line(frame, (margin, h - margin), (margin, h - margin - bracket_len), color_cyan, 2)
        # Bottom-right
        cv2.line(frame, (w - margin, h - margin), (w - margin - bracket_len, h - margin), color_cyan, 2)
        cv2.line(frame, (w - margin, h - margin), (w - margin, h - margin - bracket_len), color_cyan, 2)

        # Center Status Warning
        cv2.putText(frame, "NO WEBCAM SIGNAL", (w // 2 - 130, h // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 180, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "TEST LAB & UPLOAD MODE ACTIVE", (w // 2 - 170, h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        return frame

    def get_raw_frame(self) -> Optional[np.ndarray]:
        """Thread-safe copy of the latest raw camera frame."""
        with self.lock:
            if self.latest_raw_frame is None:
                return None
            return self.latest_raw_frame.copy()

    def release(self):
        """Cleanly terminates capture thread and releases camera device."""
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.is_available = False
        print("[CameraStreamer] Camera device released cleanly.")


class AsyncDetectionWorker:
    """
    Dedicated worker thread running YOLO object detection on the latest frame.
    Computes real detection FPS, real inference latency, and bounding box annotations
    with proximity estimated from box size (never claiming 'approaching').
    """

    def __init__(self, camera_streamer: CameraStreamer, conf_threshold: float = 0.25, imgsz: int = 416):
        self.camera_streamer = camera_streamer
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.running = False
        self.lock = threading.Lock()

        self.latest_annotated_frame: Optional[np.ndarray] = None
        self.latest_detections: List[Dict[str, Any]] = []
        self.detection_fps = 0.0
        self.detection_latency_ms = 0.0
        self._det_count = 0
        self._last_fps_calc = time.perf_counter()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(
            target=self._detection_loop,
            name="AsyncDetectionThread",
            daemon=True
        )
        self._thread.start()

    def _detection_loop(self):
        while self.running:
            frame = self.camera_streamer.get_raw_frame()
            if frame is None:
                time.sleep(0.03)
                continue

            t_start = time.perf_counter()

            # Run YOLO with dedicated YOLO_LOCK
            try:
                with YOLO_LOCK:
                    results = yolo_model.predict(
                        frame,
                        conf=self.conf_threshold,
                        imgsz=self.imgsz,
                        verbose=False
                    )

                detections, annotated = self._process_and_annotate(frame, results)
                dt_ms = (time.perf_counter() - t_start) * 1000.0

                with self.lock:
                    self.latest_annotated_frame = annotated
                    self.latest_detections = detections
                    self.detection_latency_ms = round(dt_ms, 1)

                self._update_detection_fps()
            except Exception as e:
                print(f"[AsyncDetectionWorker] Inference error: {e}")
                time.sleep(0.05)

            # Prevent CPU thrashing if detection is extremely fast
            time.sleep(0.005)

    def _update_detection_fps(self):
        self._det_count += 1
        now = time.perf_counter()
        dt = now - self._last_fps_calc
        if dt >= 1.0:
            instant_fps = self._det_count / dt
            self.detection_fps = round(0.7 * self.detection_fps + 0.3 * instant_fps, 1) if self.detection_fps > 0 else round(instant_fps, 1)
            self._det_count = 0
            self._last_fps_calc = now

    def _process_and_annotate(
        self,
        frame: np.ndarray,
        results
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        annotated = frame.copy()
        img_h, img_w = frame.shape[:2]
        detections: List[Dict[str, Any]] = []

        if not results:
            return detections, annotated

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return detections, annotated

        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = float(box.conf[0].cpu())
            cls_id = int(box.cls[0].cpu())
            class_name = res.names.get(cls_id, f"obj_{cls_id}")

            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1

            # Spatial horizontal zone
            norm_x = x_center / img_w
            if norm_x < 0.33:
                pos = "left"
            elif norm_x < 0.66:
                pos = "center"
            else:
                pos = "right"

            # Proximity estimated honestly from bounding box height ratio
            height_ratio = h / img_h
            if height_ratio >= 0.45:
                proximity = "Close"
            elif height_ratio >= 0.22:
                proximity = "Nearby"
            else:
                proximity = "Far"

            det_dict = {
                "object": class_name,
                "confidence": round(conf, 2),
                "position": pos,
                "proximity": proximity,
                "bbox": [x1, y1, x2, y2],
                "x_center": x_center,
                "y_center": y_center,
                "width": w,
                "height": h,
                "image_width": img_w,
                "image_height": img_h
            }
            detections.append(det_dict)

            # High-Contrast WCAG AA Compliant HUD Bounding Box
            color_bgr = get_class_color(class_name)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)

            # HUD Corner Brackets
            corner_len = min(15, w // 4, h // 4)
            if corner_len > 4:
                # Top-left corner
                cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), 2)
                cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), 2)
                # Bottom-right corner
                cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), 2)
                cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), 2)

            # High-contrast label tag: Object + Conf% + [Proximity]
            label_text = f"{class_name.upper()} {int(conf * 100)}% [{proximity.upper()}]"
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            )
            tag_y1 = max(0, y1 - text_h - 6)
            tag_y2 = y1

            # Filled tag background
            cv2.rectangle(annotated, (x1, tag_y1), (x1 + text_w + 6, tag_y2), (15, 23, 42), -1)
            cv2.rectangle(annotated, (x1, tag_y1), (x1 + text_w + 6, tag_y2), color_bgr, 1)

            # Label text
            cv2.putText(
                annotated,
                label_text,
                (x1 + 3, tag_y2 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return detections, annotated

    def get_latest_data(self) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]], float, float]:
        with self.lock:
            frame_copy = self.latest_annotated_frame.copy() if self.latest_annotated_frame is not None else None
            return (
                frame_copy,
                list(self.latest_detections),
                self.detection_fps,
                self.detection_latency_ms
            )

    def stop(self):
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        print("[AsyncDetectionWorker] Detection worker stopped cleanly.")

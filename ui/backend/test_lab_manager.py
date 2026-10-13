"""
Test Lab Subsystem Manager for AI Vision Assistant (Phase 5).
Handles:
- Media upload with type and size validation (localhost only)
- Automatic cleanup of temporary upload files
- Sample asset picker (test_scene.jpg, door_test.png, ocr_test.png, test_video.mp4)
- Static image YOLO detection annotation with bounding boxes and confidence
- Asynchronous 'Analyse movement' runner via real process_video() with live frame progress
- Test Lab query execution (voice, typed, chips) on the active media
- Dynamic evaluation summary loader from experiment_02_summary.json (nothing hard-coded)
"""

import os
import sys
import time
import json
import shutil
import threading
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.assistant_pipeline import yolo_model, detect_objects, process_query
from src.pipeline.model_lock import YOLO_LOCK
from src.pipeline.video_assistant_pipeline import process_video
from src.voice.tts_worker import get_tts_worker

UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "uploads")
SCRATCH_DIR = os.path.join(PROJECT_ROOT, "scratch")
SAMPLE_DIR = os.path.join(PROJECT_ROOT, "data", "test")

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB limit


class TestLabManager:
    """Manages Test Lab media, movement analysis, and evaluation metrics."""

    def __init__(self):
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        os.makedirs(SCRATCH_DIR, exist_ok=True)

        self.lock = threading.Lock()
        self.tts_worker = get_tts_worker()

        # Currently loaded media state
        self.active_media = {
            "name": "door_test.png",
            "path": os.path.join(SAMPLE_DIR, "door_test.png"),
            "type": "image",
            "size_bytes": 0,
            "detections": [],
            "annotated_image_path": None,
            "width": 0,
            "height": 0
        }

        # Movement analysis background job state
        self.analysis_job = {
            "status": "idle",  # "idle" | "running" | "completed" | "error"
            "current_frame": 0,
            "total_frames": 0,
            "percent": 0.0,
            "elapsed_sec": 0.0,
            "events": [],
            "error_message": "",
            "start_time": 0.0
        }
        self._analysis_thread: Optional[threading.Thread] = None

        # Initialize with default sample
        self.select_sample("door_test.png")

    def cleanup_temp_uploads(self):
        """Deletes all temporary upload files in data/uploads/."""
        with self.lock:
            if os.path.exists(UPLOAD_DIR):
                for filename in os.listdir(UPLOAD_DIR):
                    file_path = os.path.join(UPLOAD_DIR, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        print(f"[TestLab] Cleanup error for {file_path}: {e}")

    def get_available_samples(self) -> List[Dict[str, Any]]:
        """Returns catalog of verified test sample files."""
        samples = [
            {
                "filename": "test_scene.jpg",
                "label": "Living Room Scene",
                "type": "image",
                "description": "Indoor living room scene with furniture, chairs, tables, and objects."
            },
            {
                "filename": "door_test.png",
                "label": "Doorway Entrance",
                "type": "image",
                "description": "Clear doorway passage testing custom DeepDoors2 door detection."
            },
            {
                "filename": "ocr_test.png",
                "label": "Text Signage (OCR)",
                "type": "image",
                "description": "Signage and text card for visual OCR text reading verification."
            },
            {
                "filename": "test_video.mp4",
                "label": "Dynamic Movement Video",
                "type": "video",
                "description": "Continuous walking video with moving persons, obstacles, and directional motion."
            }
        ]

        # Verify existence and file sizes
        for s in samples:
            path = os.path.join(SAMPLE_DIR, s["filename"])
            s["available"] = os.path.exists(path)
            s["size_bytes"] = os.path.getsize(path) if s["available"] else 0

        return samples

    def select_sample(self, filename: str) -> Dict[str, Any]:
        """Loads a chosen sample asset into the Test Lab."""
        sample_path = os.path.join(SAMPLE_DIR, filename)
        if not os.path.exists(sample_path):
            raise FileNotFoundError(f"Sample file not found: {filename}")

        ext = os.path.splitext(filename)[1].lower()
        media_type = "video" if ext in ALLOWED_VIDEO_EXTENSIONS else "image"

        return self._set_active_media(sample_path, filename, media_type)

    def handle_upload(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Validates and saves uploaded file, removing previous temporary files."""
        ext = os.path.splitext(filename)[1].lower()

        # Validate file extension
        if ext not in ALLOWED_IMAGE_EXTENSIONS and ext not in ALLOWED_VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported file type '{ext}'. Allowed: images ({', '.join(ALLOWED_IMAGE_EXTENSIONS)}) or videos ({', '.join(ALLOWED_VIDEO_EXTENSIONS)}).")

        # Validate file size
        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            raise ValueError(f"File size exceeds maximum limit of {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB.")

        if len(file_bytes) == 0:
            raise ValueError("Uploaded file is empty.")

        # Clean previous uploads to prevent disk buildup
        self.cleanup_temp_uploads()

        # Sanitize filename and save to data/uploads
        safe_name = os.path.basename(filename).replace(" ", "_")
        dest_path = os.path.join(UPLOAD_DIR, safe_name)
        with open(dest_path, "wb") as f:
            f.write(file_bytes)

        media_type = "video" if ext in ALLOWED_VIDEO_EXTENSIONS else "image"
        return self._set_active_media(dest_path, safe_name, media_type)

    def _set_active_media(self, path: str, name: str, media_type: str) -> Dict[str, Any]:
        """Sets active media and annotates if image."""
        with self.lock:
            size_bytes = os.path.getsize(path) if os.path.exists(path) else 0
            self.active_media = {
                "name": name,
                "path": path,
                "type": media_type,
                "size_bytes": size_bytes,
                "detections": [],
                "annotated_image_path": None,
                "width": 0,
                "height": 0
            }

            if media_type == "image":
                self._annotate_active_image()
            elif media_type == "video":
                cap = cv2.VideoCapture(path)
                if cap.isOpened():
                    self.active_media["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.active_media["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self.active_media["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.active_media["fps"] = round(cap.get(cv2.CAP_PROP_FPS) or 30.0, 1)
                    cap.release()

            return dict(self.active_media)

    def _annotate_active_image(self):
        """Runs YOLO on the active image and renders bounding boxes with confidence."""
        img_path = self.active_media["path"]
        img = cv2.imread(img_path)
        if img is None:
            return

        h, w = img.shape[:2]
        self.active_media["width"] = w
        self.active_media["height"] = h

        # Detect objects
        detections = detect_objects(img_path)
        self.active_media["detections"] = detections

        # Draw boxes and confidence on image
        annotated = img.copy()
        for det in detections:
            bbox = det.get("bbox")
            obj_name = det.get("object", "object").upper()
            conf = det.get("confidence", 0.0)
            prox = det.get("proximity", "")
            pos = det.get("position", "")

            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
            else:
                xc = int(det.get("x_center", w // 2) * w)
                yc = int(det.get("y_center", h // 2) * h)
                bw = int(det.get("width", 50) * w)
                bh = int(det.get("height", 50) * h)
                x1, y1, x2, y2 = xc - bw // 2, yc - bh // 2, xc + bw // 2, yc + bh // 2

            # Theme-aligned bounding box (warm amber for door, safe green for other objects)
            color = (75, 184, 242) if "DOOR" in obj_name else (137, 211, 123)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{obj_name} {conf:.2f} ({pos}, {prox})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - 20)), (x1 + tw + 6, max(20, y1)), (64, 39, 30), -1)
            cv2.putText(annotated, label, (x1 + 3, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        annotated_path = os.path.join(SCRATCH_DIR, "test_lab_annotated.jpg")
        cv2.imwrite(annotated_path, annotated)
        self.active_media["annotated_image_path"] = annotated_path

    def get_active_media_info(self) -> Dict[str, Any]:
        with self.lock:
            info = dict(self.active_media)
            # Do not expose full absolute path in UI telemetry
            info["display_path"] = os.path.basename(info.get("path", ""))
            return info

    def start_movement_analysis(self) -> Dict[str, Any]:
        """Runs the real process_video() on the active video file in background."""
        with self.lock:
            if self.active_media["type"] != "video":
                raise ValueError("Active media is not a video. Load a video file to run movement analysis.")

            if self.analysis_job["status"] == "running":
                return dict(self.analysis_job)

            video_path = self.active_media["path"]
            self.analysis_job = {
                "status": "running",
                "current_frame": 0,
                "total_frames": 0,
                "percent": 0.0,
                "elapsed_sec": 0.0,
                "events": [],
                "error_message": "",
                "start_time": time.time(),
                "video_name": self.active_media["name"]
            }

        def _worker():
            t_start = time.perf_counter()

            def progress_cb(current, total):
                with self.lock:
                    self.analysis_job["current_frame"] = current
                    self.analysis_job["total_frames"] = total
                    self.analysis_job["percent"] = round((current / max(1, total)) * 100.0, 1)
                    self.analysis_job["elapsed_sec"] = round(time.perf_counter() - t_start, 1)

            try:
                events = process_video(video_path, progress_callback=progress_cb)
                with self.lock:
                    self.analysis_job["status"] = "completed"
                    self.analysis_job["events"] = events
                    self.analysis_job["percent"] = 100.0
                    self.analysis_job["elapsed_sec"] = round(time.perf_counter() - t_start, 2)
            except Exception as e:
                print(f"[TestLab] Movement analysis error: {e}")
                with self.lock:
                    self.analysis_job["status"] = "error"
                    self.analysis_job["error_message"] = str(e)
                    self.analysis_job["elapsed_sec"] = round(time.perf_counter() - t_start, 2)

        self._analysis_thread = threading.Thread(
            target=_worker,
            name="MovementAnalysisThread",
            daemon=True
        )
        self._analysis_thread.start()

        return dict(self.analysis_job)

    def get_analysis_progress(self) -> Dict[str, Any]:
        with self.lock:
            if self.analysis_job["status"] == "running":
                self.analysis_job["elapsed_sec"] = round(time.time() - self.analysis_job["start_time"], 1)
            return dict(self.analysis_job)

    def execute_query(self, query: str, speak: bool = True) -> Dict[str, Any]:
        """
        Executes assistant query on the currently loaded Test Lab media.
        Supports voice, typed queries, and chips.
        """
        if self.tts_worker:
            self.tts_worker.stop_query_speech()

        with self.lock:
            media = dict(self.active_media)

        t_start = time.perf_counter()

        if media["type"] == "image":
            target_image = media["path"]
        else:
            # For video, extract first/representative frame for the query
            cap = cv2.VideoCapture(media["path"])
            frame_extracted = None
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame_extracted = frame
                cap.release()

            target_image = os.path.join(SCRATCH_DIR, "test_lab_video_query_frame.jpg")
            if frame_extracted is not None:
                cv2.imwrite(target_image, frame_extracted)
            else:
                target_image = os.path.join(SAMPLE_DIR, "door_test.png")

        # Execute full assistant pipeline
        result = process_query(query, target_image)
        total_time_ms = round((time.perf_counter() - t_start) * 1000.0, 1)

        # Speak answer using priority 5 (user response)
        response_text = result.get("response", "")
        if response_text and speak and self.tts_worker:
            self.tts_worker.speak(response_text, priority=5)

        result["total_elapsed_ms"] = total_time_ms
        return result

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Reads experiment_02_summary.json and dynamically returns evaluation data.
        Nothing hard-coded.
        """
        summary_path = os.path.join(PROJECT_ROOT, "backup", "Experiment_02_Backup", "experiment_02_summary.json")
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"Evaluation summary file not found: {summary_path}")

        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Scan for CNN architecture experiments
        cnn_dir = os.path.join(PROJECT_ROOT, "src", "cnn", "architectures")
        cnn_models = []
        if os.path.exists(cnn_dir):
            for file in sorted(os.listdir(cnn_dir)):
                if file.endswith(".py") and not file.startswith("__") and not file.startswith("test_"):
                    model_name = file.replace(".py", "").upper()
                    cnn_models.append(model_name)

        data["cnn_architectures_implemented"] = cnn_models
        data["read_timestamp"] = time.time()
        return data


_test_lab_manager: Optional[TestLabManager] = None


def get_test_lab_manager() -> TestLabManager:
    global _test_lab_manager
    if _test_lab_manager is None:
        _test_lab_manager = TestLabManager()
    return _test_lab_manager

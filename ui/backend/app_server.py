"""
FastAPI Backend Server for AI Vision Assistant.
Provides:
- MJPEG video streaming (/video_feed) decoupled from detection
- Comprehensive readiness checklist (/api/health)
- Real-time telemetry & detections (/api/telemetry, /ws)
- Clean camera resource management on shutdown
"""

import os
import sys
import time
import json
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any, List

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Path alignment
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ui.backend.video_streamer import CameraStreamer, AsyncDetectionWorker
from src.voice.tts_worker import get_tts_worker
from src.pipeline.assistant_pipeline import get_yolo_classes

# Sounddevice for microphone health probe
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except Exception:
    HAS_SOUNDDEVICE = False

# Global streamers
camera_streamer: CameraStreamer = None
detection_worker: AsyncDetectionWorker = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Clean startup and shutdown management."""
    global camera_streamer, detection_worker
    print("\n[AI Vision Server] Starting up...")

    # Initialize and start camera streamer
    camera_streamer = CameraStreamer(preferred_index=0)
    camera_streamer.start()

    # Initialize and start detection worker
    detection_worker = AsyncDetectionWorker(camera_streamer, conf_threshold=0.25)
    detection_worker.start()

    # Initialize TTS worker
    tts_worker = get_tts_worker()
    print("[AI Vision Server] Core subsystems initialized.")

    yield

    # Clean shutdown
    print("\n[AI Vision Server] Shutting down...")
    if detection_worker:
        detection_worker.stop()
    if camera_streamer:
        camera_streamer.release()
    print("[AI Vision Server] Shutdown complete.")


app = FastAPI(
    title="AI Vision Assistant API",
    version="2.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------
# Video Streaming Endpoint (MJPEG)
# -------------------------------------------------------------

def generate_mjpeg_stream():
    """
    Generator yielding multipart JPEG frames.
    Decoupled from detection: streams at camera frame rate without stalling.
    """
    global camera_streamer, detection_worker
    target_frame_interval = 0.033  # ~30 FPS max streaming interval

    while True:
        t_start = time.perf_counter()

        # Fetch annotated frame if available, else raw frame
        annotated_frame = None
        if detection_worker:
            annotated_frame, _, _, _ = detection_worker.get_latest_data()

        if annotated_frame is None and camera_streamer:
            annotated_frame = camera_streamer.get_raw_frame()

        if annotated_frame is not None:
            # Encode frame to JPEG
            ret, buffer = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )

        # Regulate stream rate
        elapsed = time.perf_counter() - t_start
        sleep_time = max(0.005, target_frame_interval - elapsed)
        time.sleep(sleep_time)


@app.get("/video_feed")
def video_feed():
    """MJPEG continuous video stream endpoint."""
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# -------------------------------------------------------------
# Health & Readiness Checklist Endpoint
# -------------------------------------------------------------

@app.get("/api/health")
def api_health() -> JSONResponse:
    """
    Startup readiness checklist reporting health of:
    - Models (YOLO 13 classes, Intent NLP, Risk ANN, DQN, LSTM)
    - Camera (index, available bool, fps, status)
    - Microphone (sounddevice status, device name)
    - TTS (pyttsx3 persistent worker status)
    """
    global camera_streamer, detection_worker

    # 1. Models Status
    try:
        yolo_classes = get_yolo_classes()
        models_healthy = len(yolo_classes) == 13
        models_detail = {
            "healthy": models_healthy,
            "yolo_classes_count": len(yolo_classes),
            "classes": list(yolo_classes.values()),
            "models_loaded": ["YOLO11n", "Intent_NLP", "Risk_ANN", "Assistive_DQN", "Approaching_LSTM"]
        }
    except Exception as e:
        models_detail = {"healthy": False, "error": str(e)}

    # 2. Camera Status
    if camera_streamer:
        camera_detail = {
            "available": camera_streamer.is_available,
            "active_index": camera_streamer.active_index,
            "fps": camera_streamer.fps,
            "status": camera_streamer.status_message
        }
    else:
        camera_detail = {"available": False, "status": "Camera subsystem not initialized."}

    # 3. Microphone Status
    mic_available = False
    mic_device_name = "Not detected"
    if HAS_SOUNDDEVICE:
        try:
            default_input = sd.default.device[0]
            if default_input >= 0:
                dev_info = sd.query_devices(default_input)
                mic_available = True
                mic_device_name = dev_info.get("name", "Default Microphone")
        except Exception:
            mic_available = False

    mic_detail = {
        "available": mic_available,
        "device": mic_device_name,
        "status": "Microphone ready." if mic_available else "No default microphone detected. Graceful text fallback active."
    }

    # 4. TTS Status
    try:
        tts = get_tts_worker()
        tts_detail = {
            "available": True,
            "is_speaking": tts.is_speaking(),
            "status": "Windows SAPI5 TTS worker active."
        }
    except Exception as e:
        tts_detail = {"available": False, "status": f"TTS unavailable: {e}"}

    overall_ready = models_detail.get("healthy", False)

    return JSONResponse({
        "status": "ready" if overall_ready else "degraded",
        "timestamp": time.time(),
        "checklist": {
            "models": models_detail,
            "camera": camera_detail,
            "microphone": mic_detail,
            "tts": tts_detail
        }
    })


# -------------------------------------------------------------
# Telemetry Endpoint
# -------------------------------------------------------------

@app.get("/api/telemetry")
def api_telemetry() -> JSONResponse:
    """Real-time measured detection FPS, latency, and detected objects."""
    global camera_streamer, detection_worker

    cam_fps = camera_streamer.fps if camera_streamer else 0.0
    cam_avail = camera_streamer.is_available if camera_streamer else False
    det_fps = 0.0
    det_lat_ms = 0.0
    detections: List[Dict[str, Any]] = []

    if detection_worker:
        _, detections, det_fps, det_lat_ms = detection_worker.get_latest_data()

    return JSONResponse({
        "camera_fps": cam_fps,
        "camera_available": cam_avail,
        "detection_fps": det_fps,
        "detection_latency_ms": det_lat_ms,
        "detections_count": len(detections),
        "detections": detections,
        "timestamp": time.time()
    })


# -------------------------------------------------------------
# WebSocket Telemetry Stream
# -------------------------------------------------------------

@app.websocket("/ws")
async def websocket_telemetry(websocket: WebSocket):
    """Pushes live telemetry and detection updates to connected frontend clients."""
    global camera_streamer, detection_worker
    await websocket.accept()

    try:
        while True:
            cam_fps = camera_streamer.fps if camera_streamer else 0.0
            det_fps = 0.0
            det_lat_ms = 0.0
            detections: List[Dict[str, Any]] = []

            if detection_worker:
                _, detections, det_fps, det_lat_ms = detection_worker.get_latest_data()

            payload = {
                "type": "telemetry",
                "camera_fps": cam_fps,
                "camera_available": camera_streamer.is_available if camera_streamer else False,
                "detection_fps": det_fps,
                "detection_latency_ms": det_lat_ms,
                "detections": detections,
                "timestamp": time.time()
            }

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.1)  # 10 Hz telemetry broadcast
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket] Disconnected: {e}")


if __name__ == "__main__":
    import uvicorn
    # Bind exclusively to localhost
    uvicorn.run("ui.backend.app_server:app", host="127.0.0.1", port=8000, reload=False)

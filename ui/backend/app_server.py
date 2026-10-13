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
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Path alignment
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ui.backend.video_streamer import CameraStreamer, AsyncDetectionWorker
from ui.backend.voice_manager import VoiceManager
from ui.backend.hazard_monitor import ProactiveHazardMonitor
from ui.backend.test_lab_manager import get_test_lab_manager
from src.voice.tts_worker import get_tts_worker
from src.pipeline.assistant_pipeline import get_yolo_classes, process_query

# Sounddevice for microphone health probe
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except Exception:
    HAS_SOUNDDEVICE = False

# Global subsystems
camera_streamer: Optional[CameraStreamer] = None
detection_worker: Optional[AsyncDetectionWorker] = None
voice_manager: Optional[VoiceManager] = None
hazard_monitor: Optional[ProactiveHazardMonitor] = None

# Active WebSocket connections
active_websockets: List[WebSocket] = []
websocket_lock = asyncio.Lock()

# Assistant state
assistant_state = {
    "status": "STANDBY",
    "active_mode": "walk",  # "walk" | "test_lab"
    "read_aloud": True,
    "last_heard": "",
    "last_query": "",
    "last_response": "",
    "last_result": None,
    "time_to_answer_ms": 0.0
}


async def broadcast_ws(payload: dict):
    """Broadcasts JSON payload to all active WebSockets."""
    async with websocket_lock:
        to_remove = []
        for ws in active_websockets:
            try:
                await ws.send_text(json.dumps(payload))
            except Exception:
                to_remove.append(ws)
        for ws in to_remove:
            if ws in active_websockets:
                active_websockets.remove(ws)


def sync_broadcast(payload: dict):
    """Thread-safe event broadcast scheduler for background threads."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(broadcast_ws(payload), loop)
    except Exception:
        pass


def on_voice_state(state: str, data: Dict[str, Any]):
    """Callback from VoiceManager state transitions."""
    assistant_state["status"] = state
    if "text" in data:
        assistant_state["last_heard"] = data["text"]
    sync_broadcast({
        "type": "voice_state",
        "state": state,
        "data": data,
        "timestamp": time.time()
    })


def on_voice_query(query: str, speak: Optional[bool] = None):
    """Callback when a speech query is recognized (wake phrase, Spacebar, or direct query)."""
    # Stop prior query utterance so answers never overlap or queue up
    tts = get_tts_worker()
    if tts:
        tts.stop_query_speech()

    # Determine whether read aloud is active
    if speak is None:
        speak = assistant_state.get("read_aloud", True)

    t_end_speech = time.perf_counter()
    assistant_state["last_query"] = query
    assistant_state["status"] = "ANALYZING"
    sync_broadcast({"type": "voice_query", "query": query, "timestamp": time.time()})

    def _on_done():
        assistant_state["status"] = "STANDBY"
        sync_broadcast({"type": "voice_state", "state": "STANDBY", "data": {}, "timestamp": time.time()})

    test_lab = get_test_lab_manager()
    if assistant_state.get("active_mode") == "test_lab":
        # Execute query against active Test Lab media
        result, tts_spoken = test_lab.execute_query(query, speak=speak, on_done=_on_done)
        time_to_answer_ms = result.get("total_elapsed_ms", 0.0)
        if tts_spoken:
            assistant_state["status"] = "SPEAKING"
            sync_broadcast({"type": "voice_state", "state": "SPEAKING", "data": {}, "timestamp": time.time()})
        else:
            _on_done()
    else:
        # Execute assistant pipeline on latest live camera frame
        frame = None
        if camera_streamer:
            frame = camera_streamer.get_raw_frame()

        # Save temporary query frame
        frame_path = os.path.join(PROJECT_ROOT, "data", "test", "live_query_frame.jpg")
        os.makedirs(os.path.dirname(frame_path), exist_ok=True)
        if frame is not None:
            cv2.imwrite(frame_path, frame)
        else:
            frame_path = os.path.join(PROJECT_ROOT, "data", "test", "door_test.png")

        result = process_query(query, frame_path)
        t_answer_ready = time.perf_counter()
        time_to_answer_ms = round((t_answer_ready - t_end_speech) * 1000.0, 1)

        # Speak response via TTS worker (priority 5 = user response)
        tts_spoken = False
        if speak and tts and tts.is_available():
            assistant_state["status"] = "SPEAKING"
            sync_broadcast({"type": "voice_state", "state": "SPEAKING", "data": {}, "timestamp": time.time()})
            tts_spoken = bool(tts.speak(result.get("response", ""), priority=5, on_done=_on_done))
        
        if not tts_spoken:
            _on_done()

    assistant_state["last_result"] = result
    assistant_state["last_response"] = result.get("response", "")
    assistant_state["time_to_answer_ms"] = time_to_answer_ms

    sync_broadcast({
        "type": "assistant_response",
        "query": query,
        "response": result.get("response", ""),
        "result": result,
        "time_to_answer_ms": time_to_answer_ms,
        "tts_spoken": tts_spoken,
        "timestamp": time.time()
    })
    return result, tts_spoken


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Clean startup and shutdown management."""
    global camera_streamer, detection_worker, voice_manager, hazard_monitor
    print("\n[AI Vision Server] Starting up...")

    # Initialize and start camera streamer
    camera_streamer = CameraStreamer(preferred_index=0)
    camera_streamer.start()

    # Initialize and start detection worker (using imgsz=416 for 57% throughput boost)
    detection_worker = AsyncDetectionWorker(camera_streamer, conf_threshold=0.25, imgsz=416)
    detection_worker.start()

    # Initialize VoiceManager
    voice_manager = VoiceManager(
        on_query_callback=on_voice_query,
        on_state_callback=on_voice_state,
        model_size="base.en"
    )
    voice_manager.start()

    # Initialize TTS worker
    get_tts_worker()

    # Initialize Proactive Hazard Monitor (Risk ANN + DQN @ 2 Hz)
    def get_latest_det():
        if detection_worker:
            _, dets, det_fps, _, frame_id, det_ts = detection_worker.get_latest_data()
            return dets, det_fps, frame_id, det_ts
        return [], 0.0, 0, 0.0

    def get_status():
        return assistant_state.get("status", "STANDBY")

    hazard_monitor = ProactiveHazardMonitor(
        get_latest_detections_fn=get_latest_det,
        get_assistant_status_fn=get_status,
        broadcast_fn=sync_broadcast
    )
    hazard_monitor.start()

    print("[AI Vision Server] Core subsystems initialized.")

    yield

    # Clean shutdown
    print("\n[AI Vision Server] Shutting down...")
    if hazard_monitor:
        hazard_monitor.stop()
    if voice_manager:
        voice_manager.stop()
    if detection_worker:
        detection_worker.stop()
    if camera_streamer:
        camera_streamer.release()
    get_test_lab_manager().cleanup_temp_uploads()
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

# Static UI Files Mount
static_dir = os.path.join(PROJECT_ROOT, "ui", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def get_index():
    """Serves the Cyber-Aura obsidian HUD single page application."""
    index_path = os.path.join(PROJECT_ROOT, "ui", "static", "index.html")
    return FileResponse(index_path)


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
            annotated_frame, *_ = detection_worker.get_latest_data()

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
        _, detections, det_fps, det_lat_ms, *_ = detection_worker.get_latest_data()

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
# Voice & Query Interaction Endpoints
# -------------------------------------------------------------

@app.post("/api/voice/trigger_space")
def api_trigger_space() -> JSONResponse:
    """Triggers Spacebar listening mode (fallback without wake phrase)."""
    global voice_manager
    if voice_manager:
        voice_manager.trigger_spacebar()
        return JSONResponse({"status": "listening", "mode": "spacebar"})
    return JSONResponse({"status": "unavailable", "message": "Voice manager not initialized"}, status_code=503)


@app.post("/api/query")
async def api_query(payload: Dict[str, Any]) -> JSONResponse:
    """Processes typed questions or quick prompt chips."""
    query = payload.get("query", "").strip()
    read_aloud_raw = payload.get("read_aloud", True)
    if isinstance(read_aloud_raw, str):
        read_aloud = read_aloud_raw.lower() not in ("false", "0", "no")
    else:
        read_aloud = bool(read_aloud_raw)

    if not query:
        return JSONResponse({"error": "Empty query"}, status_code=400)

    # Execute query through on_voice_query handler
    result, tts_spoken = on_voice_query(query, speak=read_aloud)
    return JSONResponse({
        "status": "success",
        "query": query,
        "response": assistant_state.get("last_response", ""),
        "time_to_answer_ms": assistant_state.get("time_to_answer_ms", 0.0),
        "result": assistant_state.get("last_result", {}),
        "tts_spoken": tts_spoken
    })


@app.post("/api/tts/stop")
def api_tts_stop() -> JSONResponse:
    """Stops current query speech utterance and clears lower-priority speech."""
    try:
        tts = get_tts_worker()
        if tts:
            tts.stop_query_speech()
        return JSONResponse({"status": "stopped"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/tts/speak")
def api_tts_speak(payload: Dict[str, Any]) -> JSONResponse:
    """Speaks or replays utterance through server-side TTS worker (priority 5)."""
    text = payload.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "Empty text"}, status_code=400)
    try:
        tts = get_tts_worker()
        if not tts or not tts.is_available():
            return JSONResponse({"status": "unavailable", "message": "TTS worker not available"}, status_code=503)
        tts.stop_query_speech()

        def _on_done():
            assistant_state["status"] = "STANDBY"
            sync_broadcast({"type": "voice_state", "state": "STANDBY", "data": {}, "timestamp": time.time()})

        assistant_state["status"] = "SPEAKING"
        sync_broadcast({"type": "voice_state", "state": "SPEAKING", "data": {}, "timestamp": time.time()})
        success = tts.speak(text, priority=5, on_done=_on_done)
        if not success:
            _on_done()
            return JSONResponse({"status": "error", "message": "Failed to enqueue speech"}, status_code=500)
        return JSONResponse({"status": "success", "text": text, "tts_spoken": True})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/tts/set_read_aloud")
def api_set_read_aloud(payload: Dict[str, Any]) -> JSONResponse:
    """Updates backend read_aloud flag synchronized with UI toggle."""
    raw_val = payload.get("enabled", True)
    if isinstance(raw_val, str):
        enabled = raw_val.lower() not in ("false", "0", "no")
    else:
        enabled = bool(raw_val)
    assistant_state["read_aloud"] = enabled
    return JSONResponse({"status": "success", "read_aloud": enabled})


@app.get("/api/tts/status")
def api_tts_status() -> JSONResponse:
    """Checks whether server-side TTS worker is available and whether it is actively speaking."""
    try:
        tts = get_tts_worker()
        available = tts is not None and tts.is_available()
        return JSONResponse({
            "status": "success",
            "available": available,
            "is_speaking": tts.is_speaking() if tts else False
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "available": False,
            "error": str(e)
        }, status_code=503)


@app.post("/api/video_source")
def api_video_source(payload: Dict[str, Any]) -> JSONResponse:
    """Switches Walk Mode video source between 'webcam' and looped 'video' file."""
    global camera_streamer
    source = payload.get("source", "webcam")
    default_video_path = os.path.join(PROJECT_ROOT, "data", "test", "test_video.mp4")
    path = payload.get("path", default_video_path)

    if camera_streamer:
        success, msg = camera_streamer.set_source(source, path)
        return JSONResponse({
            "status": "success" if success else "error",
            "message": msg,
            "source_type": camera_streamer.source_type,
            "source_path": camera_streamer.source_path
        })
    return JSONResponse({"error": "Camera streamer not initialized"}, status_code=503)


@app.get("/api/alerts/log")
def api_alerts_log() -> JSONResponse:
    """Retrieves ring-buffer log of all Risk ANN + DQN hazard decisions (including SILENT/WAIT)."""
    global hazard_monitor
    if hazard_monitor:
        return JSONResponse({
            "log": hazard_monitor.get_decision_log(),
            "total_alerts": hazard_monitor.total_alerts_count
        })
    return JSONResponse({"log": [], "total_alerts": 0})


# -------------------------------------------------------------
# Phase 5: Test Lab & Evaluation Endpoints
# -------------------------------------------------------------

@app.post("/api/set_mode")
def api_set_mode(payload: Dict[str, Any]) -> JSONResponse:
    """Sets active HUD mode between 'walk' and 'test_lab'."""
    mode = payload.get("mode", "walk")
    if mode not in ["walk", "test_lab"]:
        return JSONResponse({"error": "Invalid mode. Use 'walk' or 'test_lab'."}, status_code=400)
    assistant_state["active_mode"] = mode
    sync_broadcast({"type": "mode_change", "mode": mode, "timestamp": time.time()})
    return JSONResponse({"status": "success", "mode": mode})


@app.get("/api/test_lab/samples")
def api_test_lab_samples() -> JSONResponse:
    """Returns available sample assets for Test Lab."""
    samples = get_test_lab_manager().get_available_samples()
    return JSONResponse({"status": "success", "samples": samples})


@app.post("/api/test_lab/select_sample")
def api_test_lab_select_sample(payload: Dict[str, Any]) -> JSONResponse:
    """Loads a chosen sample asset into Test Lab."""
    global camera_streamer
    filename = payload.get("filename", "")
    if not filename:
        return JSONResponse({"error": "Missing filename"}, status_code=400)
    try:
        media = get_test_lab_manager().select_sample(filename)
        # If media is video and streamer exists, point streamer to it for continuous box display
        if media.get("type") == "video" and camera_streamer:
            camera_streamer.set_source("video", media["path"])
        return JSONResponse({"status": "success", "media": media})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=404)


@app.post("/api/test_lab/upload")
async def api_test_lab_upload(file: UploadFile = File(...)) -> JSONResponse:
    """Handles upload of image or video with validation and temp-file auto-cleanup."""
    global camera_streamer
    try:
        content = await file.read()
        media = get_test_lab_manager().handle_upload(content, file.filename)
        # If media is video, point streamer to it
        if media.get("type") == "video" and camera_streamer:
            camera_streamer.set_source("video", media["path"])
        return JSONResponse({"status": "success", "media": media})
    except ValueError as ve:
        return JSONResponse({"error": str(ve)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Upload processing failed: {str(e)}"}, status_code=500)


@app.get("/api/test_lab/current_media")
def api_test_lab_current_media() -> JSONResponse:
    """Returns metadata for the currently active Test Lab media."""
    media = get_test_lab_manager().get_active_media_info()
    return JSONResponse({"status": "success", "media": media})


@app.get("/api/test_lab/media_file")
def api_test_lab_media_file():
    """Serves the raw active image or video file for browser playback."""
    info = get_test_lab_manager().get_active_media_info()
    file_path = info.get("path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Active media file not found")
    media_type = "video/mp4" if info.get("type") == "video" else "image/jpeg"
    return FileResponse(file_path, media_type=media_type)


@app.get("/api/test_lab/annotated_image")
def api_test_lab_annotated_image():
    """Serves the active image annotated with YOLO bounding boxes and confidences."""
    info = get_test_lab_manager().get_active_media_info()
    ann_path = info.get("annotated_image_path")
    if not ann_path or not os.path.exists(ann_path):
        ann_path = info.get("path")
    if not ann_path or not os.path.exists(ann_path):
        raise HTTPException(status_code=404, detail="Annotated image not available")
    return FileResponse(ann_path, media_type="image/jpeg")


@app.post("/api/test_lab/analyze_movement")
def api_test_lab_analyze_movement() -> JSONResponse:
    """Executes the real process_video() pipeline on the loaded video with progress tracking."""
    try:
        job = get_test_lab_manager().start_movement_analysis()
        return JSONResponse({"status": "started", "job": job})
    except ValueError as ve:
        return JSONResponse({"error": str(ve)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/test_lab/movement_progress")
def api_test_lab_movement_progress() -> JSONResponse:
    """Returns current progress and detected events for movement analysis."""
    job = get_test_lab_manager().get_analysis_progress()
    return JSONResponse({"status": "success", "job": job})


@app.post("/api/test_lab/query")
def api_test_lab_query(payload: Dict[str, Any]) -> JSONResponse:
    """Executes query specifically against loaded Test Lab media."""
    query = payload.get("query", "").strip()
    read_aloud_raw = payload.get("read_aloud", True)
    if isinstance(read_aloud_raw, str):
        read_aloud = read_aloud_raw.lower() not in ("false", "0", "no")
    else:
        read_aloud = bool(read_aloud_raw)

    if not query:
        return JSONResponse({"error": "Empty query"}, status_code=400)
    try:
        def _on_done():
            assistant_state["status"] = "STANDBY"
            sync_broadcast({"type": "voice_state", "state": "STANDBY", "data": {}, "timestamp": time.time()})

        result, tts_spoken = get_test_lab_manager().execute_query(query, speak=read_aloud, on_done=_on_done)
        if tts_spoken:
            assistant_state["status"] = "SPEAKING"
            sync_broadcast({"type": "voice_state", "state": "SPEAKING", "data": {}, "timestamp": time.time()})
        else:
            _on_done()

        return JSONResponse({
            "status": "success",
            "query": query,
            "response": result.get("response", ""),
            "result": result,
            "stage_latencies": result.get("stage_latencies", {}),
            "time_to_answer_ms": result.get("total_elapsed_ms", 0.0),
            "tts_spoken": tts_spoken
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/evaluation")
def api_evaluation() -> JSONResponse:
    """Dynamically reads experiment_02_summary.json and CNN architecture metadata."""
    try:
        data = get_test_lab_manager().get_evaluation_summary()
        return JSONResponse({"status": "success", "data": data})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# -------------------------------------------------------------
# WebSocket Telemetry & Event Stream
# -------------------------------------------------------------

@app.websocket("/ws")
async def websocket_telemetry(websocket: WebSocket):
    """Bidirectional WebSocket for live telemetry, voice states, and Spacebar commands."""
    global camera_streamer, detection_worker, voice_manager
    await websocket.accept()

    async with websocket_lock:
        active_websockets.append(websocket)

    async def receive_loop():
        try:
            while True:
                msg_text = await websocket.receive_text()
                try:
                    data = json.loads(msg_text)
                    action = data.get("action")
                    if action == "spacebar" and voice_manager:
                        voice_manager.trigger_spacebar()
                    elif action == "query" and data.get("text"):
                        on_voice_query(data["text"])
                except Exception as e:
                    print(f"[WebSocket] Bad payload: {e}")
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    receive_task = asyncio.create_task(receive_loop())

    try:
        while True:
            cam_fps = camera_streamer.fps if camera_streamer else 0.0
            det_fps = 0.0
            det_lat_ms = 0.0
            detections: List[Dict[str, Any]] = []

            if detection_worker:
                _, detections, det_fps, det_lat_ms, *_ = detection_worker.get_latest_data()

            tts = get_tts_worker()
            current_status = assistant_state.get("status", "STANDBY")
            effective_status = "SPEAKING" if (tts and tts.is_speaking()) else current_status

            payload = {
                "type": "telemetry",
                "camera_fps": cam_fps,
                "camera_available": camera_streamer.is_available if camera_streamer else False,
                "detection_fps": det_fps,
                "detection_latency_ms": det_lat_ms,
                "detections": detections,
                "assistant_status": effective_status,
                "last_heard": assistant_state.get("last_heard", ""),
                "timestamp": time.time()
            }

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.1)  # 10 Hz telemetry broadcast
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket] Disconnected: {e}")
    finally:
        receive_task.cancel()
        async with websocket_lock:
            if websocket in active_websockets:
                active_websockets.remove(websocket)


if __name__ == "__main__":
    import uvicorn
    # Bind exclusively to localhost
    uvicorn.run("ui.backend.app_server:app", host="127.0.0.1", port=8000, reload=False)


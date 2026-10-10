# AI Vision Assistant — Real-Time Assistive Vision & Voice System

An intelligent, hands-free spatial perception assistant designed to aid visually impaired individuals through continuous object detection, Risk ANN evaluation, Assistive DQN decision-making, and fuzzy voice interaction.

---

## Architectural Notes & Open Limitations

### Walk Mode Movement State Limitation
- In **Walk Mode** (live webcam or video stream monitoring), user movement is fixed at **`"Stationary"`** for state feature vector construction (`movement=stationary`).
- **Impact on Assistive DQN:** The trained DQN model receives a state space where the movement feature is constant (`0`). Consequently, the DQN operates on a specific, constrained subset of its full trained state space.
- **Demo Disclosure:** This architectural decision ensures robust, non-erratic hazard alerts without requiring specialized physical motion sensors. In the offline **Test Lab** environment, dynamic movement states (`approaching`, `moving`, `stationary`) can be evaluated fully.

---

## Subsystem Architecture

1. **Continuous Video Streaming (`ui/backend/video_streamer.py`):**
   - Decoupled hardware capture loop running at native camera rate.
   - Async detection worker running YOLO11n at `imgsz=416` (57% throughput boost) with dedicated `YOLO_LOCK`.
   - Supports live source toggling between physical webcam and looped test video file (`data/test/test_video.mp4`).

2. **Hands-Free Voice Manager (`ui/backend/voice_manager.py`):**
   - Energy-based VAD with baseline ambient noise calibration.
   - Whisper STT (`base.en`) running under dedicated `WHISPER_LOCK`.
   - Fuzzy regex wake matching for *"Hey Vision"*, *"Hi Vision"*, *"Hey vison"*, *"Hey division"*, *"Hey visual"*.
   - Two-stage wake window (6.0s) and compound single-breath query support.
   - Echo suppression with 400ms post-TTS mute tail.

3. **Proactive Hazard Monitor (`ui/backend/hazard_monitor.py`):**
   - Runs Risk ANN + Assistive DQN evaluation at **2 Hz**.
   - Hysteresis: requires 2 consecutive cycles of a hazard before alerting.
   - Cooldown: 7.0 seconds per `(object_name, position)`.
   - Priority dispatch: `WARN` interrupts TTS immediately; `SPEAK` enqueues.
   - Proximity-only alert templates (zero usage of *"approaching"*).
   - Live DQN Decision Trace log streaming via WebSockets.

---

## Running the Application

```bash
# Start the FastAPI Backend & Cyber-Aura HUD Server
python -m uvicorn ui.backend.app_server:app --host 127.0.0.1 --port 8000
```
Open `http://localhost:8000` in your browser.

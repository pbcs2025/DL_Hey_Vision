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

### Option 1: Desktop 1-Click Launcher (Windows)
Double-click `run_app.bat` (or the desktop shortcut `AI Vision Assistant.bat`). This will:
1. Start the FastAPI server on `0.0.0.0:8000` (accessible from both desktop and phone).
2. Display your computer's local Wi-Fi IP address.
3. Automatically launch the application in an app window.

### Option 2: Manual Terminal Launch
```bash
# Start on 0.0.0.0 to enable both PC and mobile phone access
python -m uvicorn ui.backend.app_server:app --host 0.0.0.0 --port 8000
```
- **On this computer:** Open [http://localhost:8000](http://localhost:8000)
- **On your mobile phone (same Wi-Fi):** Open `http://<YOUR_LOCAL_IP>:8000` (e.g. `http://192.168.1.104:8000`)

---

## Mobile App & PWA Features

The application includes full **Progressive Web App (PWA)** support for mobile devices:
1. **Home Screen Installation:** Open the link in mobile Chrome or Safari and tap **"Install App"** / **"Add to Home Screen"** to install it as a standalone mobile application.
2. **Spoken Voice Output:** Integrates browser Speech Synthesis (`window.speechSynthesis`) so assistant answers and hazard alerts are automatically read aloud to visually impaired users.
3. **Mobile Test Lab:** Upload pictures directly from your mobile camera for real-time spatial analysis, obstacle risk scoring, and DQN navigation guidance.


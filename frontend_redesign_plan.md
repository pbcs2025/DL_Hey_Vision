# 🎨 AI Vision Assistant — Approved Architecture & Implementation Plan

## 1. Amendments & Architecture Decisions

### A. Real-Time Video & Metric Honesty (Amendment 1 & 7)
- **Decoupled Architecture**: Camera frame grab runs at hardware camera rate in a dedicated thread. YOLO detection runs asynchronously on the latest available frame.
- **Runtime Measurement**: The header and pipeline strip will display true, live `perf_counter()` metrics:
  - Video FPS (exponentially moving average)
  - Detection Latency (ms)
  - Pipeline Stage Latencies (ms for Camera, YOLO, NLP, Spatial, Risk, DQN, Voice)
  - Zero placeholder values or hardcoded benchmarks.

### B. Thread-Safe Model Lock & Throttling (Amendment 2)
- A central `threading.Lock` (`MODEL_INFERENCE_LOCK`) wraps all TensorFlow / Keras (NLP, Risk, DQN, LSTM), PyTorch / Ultralytics (YOLO), and Whisper model calls.
- Proactive Risk ANN + DQN evaluation in Walk Mode is throttled to **2–3 Hz** (every 350–500ms) rather than executing on every single video frame, keeping the UI responsive.

### C. Live Proactive Warnings vs. Test Lab (Amendment 3 — Architectural Decision)
- **Decision**: **Remove approaching from live warnings** and reserve approaching detection exclusively for Test Lab's "Analyse Movement" mode.
- **Rationale**: Live user-worn or handheld cameras suffer from ego-motion (walking forward causes stationary objects to expand in pixel dimensions, generating false "approaching" signals). Additionally, the LSTM requires a contiguous 30-frame track history (~2–3 seconds), which frequently breaks during rapid camera pans. 
- In **Walk Mode**, proactive warnings will focus honestly on **Spatial Obstacle Proximity & Centered Hazard Risk** (large bounding boxes in center path) evaluated by Risk ANN and DQN with `movement="Stationary"`. The full LSTM temporal tracker (`process_video`) is used where it belongs: in **Test Lab** on uploaded video recordings.

### D. Hands-Free Voice with Fuzzy Matching (Amendment 4)
- Whisper frequently outputs phonetically similar variations of *"Hey Vision"* (e.g., *"hey vision"*, *"a vision"*, *"high vision"*, *"hey visual"*, *"vision assistant"*).
- Implement normalized fuzzy string matching (token containment and thresholded Levenshtein similarity).
- Unsolicited speech without the wake phrase is strictly discarded unless the Spacebar key was pressed.

### E. Interruptible Priority TTS Worker (Amendment 5)
- `TTSWorker` maintains a priority queue (`queue.PriorityQueue`).
- High-priority `WARN` alerts interrupt any currently playing informational or scene-description speech (`engine.stop()` + queue clear + immediate warning playback).
- Repeated speak/interrupt cycles are hardened against Windows SAPI/pyttsx3 COM event issues.

### F. Startup Readiness Checklist & Graceful Degradation (Amendment 6)
- At boot, `app_server.py` runs a 4-point diagnostic:
  1. `[MODELS]` YOLO (13 classes), NLP, Risk ANN, DQN, LSTM loaded.
  2. `[CAMERA]` Webcam access verified. If unavailable ➔ Graceful fallback to Test Lab / media upload.
  3. `[MIC]` Audio input stream verified. If unavailable ➔ Graceful fallback to Spacebar / typed query / prompt chips.
  4. `[TTS]` Windows speech synthesizer initialized. If unavailable ➔ Silent text-only fallback.
- The UI displays an accessible status banner reflecting exact hardware availability.

### G. Model Names & Accessibility (Amendment 8)
- 13 classes are dynamically retrieved at startup via `yolo_model.names`.
- Accessibility adheres to **WCAG 2.1 AA** standards (minimum 24px operational font size in Walk Mode, 4.5:1 text-to-background contrast, non-color-dependent risk status text).

---

## 2. Phased Build Order

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: Backend Additions & Pipeline Instrumentation       │  <-- STOP FOR APPROVAL
├─────────────────────────────────────────────────────────────┤
│ PHASE 2: Live Video Engine & MJPEG Streaming                │
├─────────────────────────────────────────────────────────────┤
│ PHASE 3: Hands-Free Voice Subsystem (Wake Phrase + Earcons) │  <-- STOP FOR APPROVAL
├─────────────────────────────────────────────────────────────┤
│ PHASE 4: Proactive Safety System (Risk/DQN + Cooldowns)     │
├─────────────────────────────────────────────────────────────┤
│ PHASE 5: Test Lab & Dynamic Video Movement Engine           │
├─────────────────────────────────────────────────────────────┤
│ PHASE 6: UI Polish, WCAG AA Verification & Offline Assets   │
└─────────────────────────────────────────────────────────────┘
```

- Work is performed on git branch `frontend-redesign`.
- Atomic git commits after each phase.
- Verification report provided after each phase.
- Execution stops for user approval after **Phase 1** and **Phase 3**.

"""
Proactive Hazard Monitoring Engine for AI Vision Assistant (Phase 4.5).
Features:
- Evaluates Risk ANN + DQN at ~2 Hz on distinct new detection frames (frame-based hysteresis)
- Stale detection filter: ignores detections older than 1.0 second
- 7.0-second cooldown per (object_name, position)
- 4.0-second global minimum gap between ANY two proactive alerts
- Announces only the single highest-risk hazard per evaluation window
- Priority dispatch: WARN interrupts TTS immediately (priority 1); SPEAK enqueues (priority 10)
- Query suppression: suppresses low/med alerts while answering user queries unless Risk is HIGH/CRITICAL
- Proximity template phrasing only (zero 'approaching' wording)
- Complete DQN trace logging for all decisions (including WAIT / SILENT)
"""

import time
import threading
from typing import Dict, Any, List, Optional, Tuple, Callable

from src.pipeline.assistant_pipeline import predict_risk, make_dqn_decision
from src.voice.tts_worker import get_tts_worker

HYSTERESIS_THRESHOLD = 2       # Must persist across >= 2 distinct new detection frames
COOLDOWN_SECONDS = 10.0         # Cooldown per hazard to target 3-6 alerts/min
GLOBAL_ALERT_GAP_SECONDS = 4.0  # Minimum gap between any two proactive alerts
EVAL_INTERVAL = 0.5             # 2 Hz evaluation rate


class ProactiveHazardMonitor:
    def __init__(
        self,
        get_latest_detections_fn: Callable[[], Any],
        get_assistant_status_fn: Callable[[], str],
        broadcast_fn: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        self.get_latest_detections_fn = get_latest_detections_fn
        self.get_assistant_status_fn = get_assistant_status_fn
        self.broadcast_fn = broadcast_fn

        self.running = False
        self._thread: Optional[threading.Thread] = None
        self.tts_worker = get_tts_worker()

        # State tracking
        self.last_evaluated_frame_id = -1
        self._fallback_frame_id = 0
        self.last_global_alert_time = 0.0
        self.persistence_counts: Dict[Tuple[str, str], int] = {}
        self.last_alert_times: Dict[Tuple[str, str], float] = {}

        # Alert log ring buffer (last 100 decisions)
        self.decision_log: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

        # Telemetry metrics
        self.total_alerts_count = 0
        self.start_timestamp = time.time()

    def start(self):
        """Starts background 2 Hz evaluation loop."""
        self.running = True
        self.start_timestamp = time.time()
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="ProactiveHazardThread",
            daemon=True
        )
        self._thread.start()
        print("[HazardMonitor] Proactive hazard evaluation loop started (2 Hz).")

    def stop(self):
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        print("[HazardMonitor] Proactive hazard monitor stopped.")

    def get_decision_log(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self.decision_log)

    def _monitor_loop(self):
        while self.running:
            t_start = time.perf_counter()
            try:
                self._evaluate_hazards()
            except Exception as e:
                print(f"[HazardMonitor] Evaluation error: {e}")

            elapsed = time.perf_counter() - t_start
            sleep_time = max(0.01, EVAL_INTERVAL - elapsed)
            time.sleep(sleep_time)

    def _evaluate_hazards(self):
        det_data = self.get_latest_detections_fn()
        if isinstance(det_data, tuple):
            if len(det_data) >= 4:
                detections, det_fps, frame_id, det_ts = det_data[:4]
            elif len(det_data) == 2:
                detections, det_fps = det_data
                self._fallback_frame_id += 1
                frame_id = self._fallback_frame_id
                det_ts = time.time()
            else:
                self._fallback_frame_id += 1
                detections, det_fps, frame_id, det_ts = [], 0.0, self._fallback_frame_id, time.time()
        else:
            self._fallback_frame_id += 1
            detections, det_fps, frame_id, det_ts = [], 0.0, self._fallback_frame_id, time.time()

        current_status = self.get_assistant_status_fn()
        now = time.time()

        # 1. Ignore stale frames older than ~1.0 second (a stale frame can never trigger an alert)
        if det_ts > 0 and (now - det_ts) > 1.0:
            return

        # 2. Advance hysteresis ONLY on distinct NEW detection frames
        is_new_frame = (frame_id != self.last_evaluated_frame_id)
        if not is_new_frame:
            return
        self.last_evaluated_frame_id = frame_id

        current_detected_keys = set()
        candidates = []

        for det in detections:
            obj_name = det.get("object", "object").lower()
            position = det.get("position", "center").lower()
            proximity = det.get("proximity", "Nearby")
            key = (obj_name, position)
            current_detected_keys.add(key)

            # Update frame-based persistence count
            self.persistence_counts[key] = self.persistence_counts.get(key, 0) + 1
            persist_count = self.persistence_counts[key]

            # Check Hysteresis Threshold (must persist for >= 2 distinct frames)
            if persist_count < HYSTERESIS_THRESHOLD:
                continue

            # Predict Risk ANN (movement fixed at 'stationary' for Walk Mode)
            risk_label, risk_conf, risk_id = predict_risk(
                detection=det,
                movement="stationary",
                ahead=(position == "center")
            )

            # Make DQN Decision
            dqn_res = make_dqn_decision(
                detection=det,
                risk_id=risk_id,
                intent="OBSTACLE_CHECK",
                movement="stationary",
                recent_alert=1 if (now - self.last_alert_times.get(key, 0.0) < COOLDOWN_SECONDS) else 0
            )

            raw_action = dqn_res.get("action", "WAIT")
            q_values = dqn_res.get("q_values", [0.0, 0.0, 0.0])

            # Check Cooldown constraint (checks both (object, position) and object-level cooldown)
            last_alert_t = max(self.last_alert_times.get(key, 0.0), self.last_alert_times.get(obj_name, 0.0))
            in_cooldown = (now - last_alert_t) < COOLDOWN_SECONDS
            cooldown_remaining = max(0.0, round(COOLDOWN_SECONDS - (now - last_alert_t), 1))

            # Check Assistant Query Suppression (suppress unless Risk is HIGH/CRITICAL)
            answering_user_query = current_status in ["ANALYZING", "SPEAKING"]
            is_high_risk = risk_label in ["HIGH", "CRITICAL"]
            suppressed = answering_user_query and not is_high_risk

            # Generate Proximity Alert Phrasing (Zero "approaching" wording)
            phrase = self._format_alert_phrase(obj_name, position, proximity)
            assert "approaching" not in phrase.lower(), "SECURITY FAILURE: 'approaching' detected in Walk Mode alert phrase!"

            candidates.append({
                "det": det,
                "key": key,
                "object": obj_name,
                "position": position,
                "proximity": proximity,
                "risk_label": risk_label,
                "risk_conf": risk_conf,
                "risk_id": risk_id,
                "raw_action": raw_action,
                "q_values": q_values,
                "in_cooldown": in_cooldown,
                "cooldown_remaining": cooldown_remaining,
                "suppressed": suppressed,
                "phrase": phrase
            })

        # Reset hysteresis for objects no longer visible
        for k in list(self.persistence_counts.keys()):
            if k not in current_detected_keys:
                self.persistence_counts[k] = 0

        # Global alert gap check (minimum 4.0 seconds between any two proactive alerts)
        in_global_gap = (now - self.last_global_alert_time) < GLOBAL_ALERT_GAP_SECONDS

        # Filter candidates eligible to speak
        eligible = [
            c for c in candidates
            if c["raw_action"] in ["WARN", "SPEAK"]
            and not c["in_cooldown"]
            and not c["suppressed"]
        ]

        # Select only the single highest-risk candidate per window
        best_candidate = None
        if eligible and not in_global_gap:
            # Sort order: risk_id desc, close proximity desc, risk_conf desc
            eligible.sort(
                key=lambda c: (
                    c["risk_id"],
                    1 if c["proximity"].lower() == "close" else 0,
                    c["risk_conf"]
                ),
                reverse=True
            )
            best_candidate = eligible[0]

        # Process effective actions and log every decision
        for c in candidates:
            key = c["key"]
            is_chosen = (best_candidate is not None and c is best_candidate)

            effective_action = "WAIT"
            spoke_alert = False

            if is_chosen:
                effective_action = c["raw_action"]
                if effective_action == "WARN":
                    # Priority 1 = urgent warning (interrupts everything)
                    self.tts_worker.interrupt_and_warn(c["phrase"])
                    self.last_alert_times[key] = now
                    self.last_alert_times[c["object"]] = now
                    self.last_global_alert_time = now
                    self.total_alerts_count += 1
                    spoke_alert = True
                elif effective_action == "SPEAK":
                    # Priority 10 = proactive speech (waits behind user query answers which use priority 5)
                    self.tts_worker.speak(c["phrase"], priority=10)
                    self.last_alert_times[key] = now
                    self.last_alert_times[c["object"]] = now
                    self.last_global_alert_time = now
                    self.total_alerts_count += 1
                    spoke_alert = True

            log_entry = {
                "timestamp": now,
                "time_str": time.strftime("%H:%M:%S", time.localtime(now)),
                "object": c["object"],
                "position": c["position"],
                "proximity": c["proximity"],
                "risk": c["risk_label"],
                "risk_confidence": round(c["risk_conf"], 3),
                "raw_action": c["raw_action"],
                "effective_action": effective_action,
                "spoke_alert": spoke_alert,
                "q_values": [round(q, 3) for q in c["q_values"]],
                "phrase": c["phrase"] if spoke_alert else "",
                "in_cooldown": c["in_cooldown"],
                "in_global_gap": in_global_gap,
                "cooldown_remaining_sec": c["cooldown_remaining"],
                "suppressed": c["suppressed"],
                "frame_id": frame_id,
                "detection_fps": round(det_fps, 1)
            }

            with self.lock:
                self.decision_log.append(log_entry)
                if len(self.decision_log) > 100:
                    self.decision_log.pop(0)

            # Broadcast live via WebSocket
            if self.broadcast_fn:
                self.broadcast_fn({
                    "type": "dqn_decision",
                    "decision": log_entry
                })

    def _format_alert_phrase(self, object_name: str, position: str, proximity: str) -> str:
        """Constructs proximity-based template phrase without the word 'approaching'."""
        obj_clean = object_name.title()
        is_close = (proximity.lower() == "close")

        if position == "center":
            return f"{obj_clean} ahead, close." if is_close else f"{obj_clean} ahead."
        elif position == "left":
            return f"{obj_clean} on the left, close." if is_close else f"{obj_clean} on your left."
        elif position == "right":
            return f"{obj_clean} on the right, close." if is_close else f"{obj_clean} on your right."
        else:
            return f"{obj_clean} nearby."

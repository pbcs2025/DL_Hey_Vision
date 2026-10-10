"""
Proactive Hazard Monitoring Engine for AI Vision Assistant (Phase 4).
Features:
- Evaluates Risk ANN + DQN at ~2 Hz on latest camera/video detections
- Hysteresis: requires 2 consecutive evaluations of a hazard before alerting
- Cooldown: 7.0 seconds per (object_name, position) key
- Priority dispatch: WARN interrupts TTS immediately; SPEAK enqueues
- Query suppression: suppresses low/med alerts while answering user queries unless Risk is HIGH/CRITICAL
- Proximity template phrasing only (zero 'approaching' wording)
- Logs every decision (including WAIT / SILENT) to ring buffer and WebSocket stream
"""

import time
import threading
from typing import Dict, Any, List, Optional, Tuple, Callable

from src.pipeline.assistant_pipeline import predict_risk, make_dqn_decision
from src.voice.tts_worker import get_tts_worker

HYSTERESIS_THRESHOLD = 2  # Must persist for 2 consecutive 2Hz cycles (1.0s)
COOLDOWN_SECONDS = 7.0     # Cooldown per (object, position)
EVAL_INTERVAL = 0.5        # 2 Hz evaluation rate


class ProactiveHazardMonitor:
    def __init__(
        self,
        get_latest_detections_fn: Callable[[], Tuple[List[Dict[str, Any]], float]],
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
        detections, det_fps = self.get_latest_detections_fn()
        current_status = self.get_assistant_status_fn()
        now = time.time()

        current_detected_keys = set()

        for det in detections:
            obj_name = det.get("object", "object").lower()
            position = det.get("position", "center").lower()
            proximity = det.get("proximity", "Nearby")
            key = (obj_name, position)
            current_detected_keys.add(key)

            # 1. Update Hysteresis Persistence Count
            self.persistence_counts[key] = self.persistence_counts.get(key, 0) + 1
            persist_count = self.persistence_counts[key]

            # 2. Check Hysteresis Threshold
            if persist_count < HYSTERESIS_THRESHOLD:
                continue

            # 3. Predict Risk ANN (movement fixed at 'stationary' for Walk Mode)
            risk_label, risk_conf, risk_id = predict_risk(
                detection=det,
                movement="stationary",
                ahead=(position == "center")
            )

            # 4. Make DQN Decision
            # intent="OBSTACLE_CHECK" for proactive scan
            dqn_res = make_dqn_decision(
                detection=det,
                risk_id=risk_id,
                intent="OBSTACLE_CHECK",
                movement="stationary",
                recent_alert=1 if (now - self.last_alert_times.get(key, 0.0) < COOLDOWN_SECONDS) else 0
            )

            raw_action = dqn_res.get("action", "WAIT")
            q_values = dqn_res.get("q_values", [0.0, 0.0, 0.0])

            # 5. Check Cooldown constraint
            last_alert_t = self.last_alert_times.get(key, 0.0)
            in_cooldown = (now - last_alert_t) < COOLDOWN_SECONDS
            cooldown_remaining = max(0.0, round(COOLDOWN_SECONDS - (now - last_alert_t), 1))

            # 6. Check Assistant Query Suppression (suppress unless Risk is HIGH/CRITICAL)
            answering_user_query = current_status in ["ANALYZING", "SPEAKING"]
            is_high_risk = risk_label in ["HIGH", "CRITICAL"]
            suppressed = answering_user_query and not is_high_risk

            effective_action = raw_action
            if in_cooldown or suppressed:
                effective_action = "WAIT"

            # 7. Generate Proximity Alert Phrasing (Zero "approaching" wording)
            phrase = self._format_alert_phrase(obj_name, position, proximity)
            assert "approaching" not in phrase.lower(), "SECURITY FAILURE: 'approaching' detected in Walk Mode alert phrase!"

            # 8. Dispatch TTS based on effective action
            spoke_alert = False
            if effective_action == "WARN":
                self.tts_worker.interrupt_and_warn(phrase)
                self.last_alert_times[key] = now
                self.total_alerts_count += 1
                spoke_alert = True
            elif effective_action == "SPEAK":
                self.tts_worker.speak(phrase, priority=5)
                self.last_alert_times[key] = now
                self.total_alerts_count += 1
                spoke_alert = True

            # 9. Record Decision Entry
            log_entry = {
                "timestamp": now,
                "time_str": time.strftime("%H:%M:%S", time.localtime(now)),
                "object": obj_name,
                "position": position,
                "proximity": proximity,
                "risk": risk_label,
                "risk_confidence": round(risk_conf, 3),
                "raw_action": raw_action,
                "effective_action": effective_action,
                "spoke_alert": spoke_alert,
                "q_values": [round(q, 3) for q in q_values],
                "phrase": phrase if spoke_alert else "",
                "in_cooldown": in_cooldown,
                "cooldown_remaining_sec": cooldown_remaining,
                "suppressed": suppressed,
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

        # Reset hysteresis for objects no longer visible
        for k in list(self.persistence_counts.keys()):
            if k not in current_detected_keys:
                self.persistence_counts[k] = 0

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

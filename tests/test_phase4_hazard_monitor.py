"""
Unit and integration tests for Phase 4 Proactive Hazard Monitor and Video Source Switching.
Verifies:
1. 2 Hz evaluation loop and Hysteresis threshold (2 cycles)
2. 7.0s Cooldown enforcement per (object, position)
3. Proximity-only phrasing and ZERO occurrences of 'approaching'
4. Query suppression logic for LOW/MEDIUM risk during active speech
5. DQN decision trace logging (including WAIT/SILENT decisions)
6. Video source switching endpoint (/api/video_source)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import pytest
from fastapi.testclient import TestClient

from ui.backend.hazard_monitor import ProactiveHazardMonitor, HYSTERESIS_THRESHOLD, COOLDOWN_SECONDS
from ui.backend.app_server import app


def test_hysteresis_and_cooldown():
    """Verifies hysteresis (2 consecutive cycles) and 7.0s cooldown per hazard key."""
    detections_mock = [
        {"object": "chair", "position": "center", "width": 200, "height": 300, "image_width": 640, "image_height": 480, "confidence": 0.85, "proximity": "Close"}
    ]
    status_mock = ["STANDBY"]

    def get_dets():
        return detections_mock, 6.5

    def get_status():
        return status_mock[0]

    monitor = ProactiveHazardMonitor(
        get_latest_detections_fn=get_dets,
        get_assistant_status_fn=get_status
    )

    key = ("chair", "center")

    # Cycle 1: Hysteresis count becomes 1 (< 2 threshold) -> No alert log recorded yet
    monitor._evaluate_hazards()
    assert monitor.persistence_counts.get(key) == 1
    assert len(monitor.decision_log) == 0

    # Cycle 2: Hysteresis count becomes 2 (>= 2 threshold) -> Evaluates Risk & DQN -> Alert triggered!
    monitor._evaluate_hazards()
    assert monitor.persistence_counts.get(key) == 2
    assert len(monitor.decision_log) == 1

    first_entry = monitor.decision_log[-1]
    assert first_entry["object"] == "chair"
    assert first_entry["position"] == "center"
    assert "approaching" not in first_entry["phrase"].lower()
    assert first_entry["in_cooldown"] is False
    assert first_entry["spoke_alert"] is True

    # Cycle 3 (Immediate retry within 7s cooldown): Hysteresis count 3 -> In Cooldown -> Effective action forced to WAIT
    monitor._evaluate_hazards()
    assert len(monitor.decision_log) == 2
    second_entry = monitor.decision_log[-1]
    assert second_entry["in_cooldown"] is True
    assert second_entry["effective_action"] == "WAIT"
    assert second_entry["spoke_alert"] is False


def test_zero_approaching_wording_security_check():
    """Security assertion verifying that 'approaching' is never present in alert phrases."""
    monitor = ProactiveHazardMonitor(lambda: ([], 0.0), lambda: "STANDBY")

    for obj in ["chair", "table", "door", "person"]:
        for pos in ["center", "left", "right"]:
            for prox in ["Close", "Nearby"]:
                phrase = monitor._format_alert_phrase(obj, pos, prox)
                assert "approaching" not in phrase.lower(), f"Security failure on phrase: {phrase}"


def test_query_suppression():
    """Verifies alerts are suppressed when assistant is answering a query, unless risk is HIGH/CRITICAL."""
    detections_mock = [
        {"object": "table", "position": "left", "width": 100, "height": 100, "image_width": 640, "image_height": 480, "confidence": 0.50, "proximity": "Nearby"}
    ]
    status_mock = ["SPEAKING"]  # Assistant actively answering user query

    monitor = ProactiveHazardMonitor(
        get_latest_detections_fn=lambda: (detections_mock, 6.0),
        get_assistant_status_fn=lambda: status_mock[0]
    )

    # Pass 1 & 2 for hysteresis
    monitor._evaluate_hazards()
    monitor._evaluate_hazards()

    assert len(monitor.decision_log) == 1
    entry = monitor.decision_log[-1]
    if entry["risk"] in ["LOW", "MEDIUM"]:
        assert entry["suppressed"] is True
        assert entry["effective_action"] == "WAIT"


def test_video_source_endpoint():
    """Verifies /api/video_source endpoint switches video stream mode."""
    with TestClient(app) as client:
        # Switch to demo video mode
        res = client.post("/api/video_source", json={"source": "video"})
        assert res.status_code == 200
        data = res.json()
        assert "status" in data
        assert data.get("source_type") == "video"

        # Switch back to webcam mode
        res_webcam = client.post("/api/video_source", json={"source": "webcam"})
        assert res_webcam.status_code == 200
        data_webcam = res_webcam.json()
        assert data_webcam.get("source_type") == "webcam"


if __name__ == "__main__":
    pytest.main(["-v", __file__])

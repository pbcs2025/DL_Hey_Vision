"""
Unit and integration tests for Phase 3 Voice Subsystem.
Verifies:
1. Fuzzy wake phrase matching (compound queries and two-stage wake words)
2. Speech without wake phrase is rejected
3. Spacebar physical fallback trigger
4. Echo control: TTS speaking muting + post-TTS tail
5. Direct typed query & chips endpoint (/api/query)
6. HTML HUD static serving and Start Assistant overlay
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import pytest
from fastapi.testclient import TestClient

from ui.backend.voice_manager import match_wake_phrase, VoiceManager, ECHO_TAIL_SECONDS
from ui.backend.app_server import app


def test_wake_phrase_fuzzy_matching():
    """Verifies that fuzzy regex correctly matches compound queries and two-stage wake words."""
    compound_queries = [
        ("Hey Vision, where is the door?", "where is the door"),
        ("Hi vision, what is in front of me?", "what is in front of me"),
        ("Hey Visual, describe the room.", "describe the room"),
        ("Hey vison, can I walk forward?", "can i walk forward"),
        ("Hey division, find the table.", "find the table"),
    ]

    for raw, expected_trailing in compound_queries:
        is_wake, trailing = match_wake_phrase(raw)
        assert is_wake is True, f"Failed to match compound wake phrase: {raw}"
        assert trailing == expected_trailing, f"Expected trailing '{expected_trailing}', got '{trailing}'"

    two_stage_phrases = [
        "Hey Vision",
        "hey vision.",
        "Hi Vision",
        "Hey visual",
        "Hey vison",
        "Hey division",
        "Hey Vision!"
    ]

    for raw in two_stage_phrases:
        is_wake, trailing = match_wake_phrase(raw)
        assert is_wake is True, f"Failed to match two-stage wake phrase: {raw}"
        assert trailing == "", f"Expected empty trailing for two-stage, got '{trailing}'"


def test_speech_without_wake_is_rejected():
    """Verifies that everyday conversational speech without wake words is ignored."""
    non_wake_phrases = [
        "hey listen, what is that object?",
        "a vision, what do you see?",
        "vision assistant, find the table.",
        "Can you pass the salt please?",
        "The weather forecast says it will rain today.",
        "I need to buy some milk and eggs tomorrow.",
        "The television is too loud in the living room.",
        "We have a project deadline coming up.",
        "He has great visual acuity with glasses.",
        "Let's review the quarterly report together.",
        "Where did I leave my car keys?",
        "What time is it right now?"
    ]

    for phrase in non_wake_phrases:
        is_wake, _ = match_wake_phrase(phrase)
        assert is_wake is False, f"False trigger on non-wake speech: {phrase}"


def test_spacebar_trigger():
    """Verifies that Spacebar immediately arms listening without wake phrase."""
    states_received = []

    def mock_state_cb(state, data):
        states_received.append((state, data))

    vm = VoiceManager(
        on_query_callback=lambda q: None,
        on_state_callback=mock_state_cb,
        model_size="base.en"
    )

    assert vm.spacebar_active is False
    assert vm.is_awake is False

    vm.trigger_spacebar()

    assert vm.spacebar_active is True
    assert vm.is_awake is True
    assert any(s[0] == "LISTENING" and s[1].get("reason") == "spacebar" for s in states_received)


def test_echo_control_tail():
    """Verifies echo control tail duration constant."""
    assert ECHO_TAIL_SECONDS == 0.4


def test_app_server_endpoints():
    """Verifies root HTML UI, trigger_space, and direct /api/query endpoint."""
    with TestClient(app) as client:
        # 1. Root HTML UI endpoint
        res = client.get("/")
        assert res.status_code == 200
        html = res.text
        assert "AI VISION ASSISTANT" in html
        assert "startOverlay" in html
        assert "unfocusedWarning" in html

        # 2. Trigger spacebar endpoint
        res_space = client.post("/api/voice/trigger_space")
        assert res_space.status_code == 200
        data_space = res_space.json()
        assert data_space.get("status") == "listening"
        assert data_space.get("mode") == "spacebar"

        # 3. Direct typed query endpoint (works even without mic)
        res_q = client.post("/api/query", json={"query": "where is the door"})
        assert res_q.status_code == 200
        data_q = res_q.json()
        assert data_q.get("status") == "success"
        assert "response" in data_q
        assert len(data_q["response"]) > 0
        assert data_q.get("time_to_answer_ms", 0) > 0


if __name__ == "__main__":
    pytest.main(["-v", __file__])

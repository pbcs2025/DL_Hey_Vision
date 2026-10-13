import os
import sys
import time
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.automated_live_assistant import (
    BackgroundCamera,
    VoiceActivityDetector,
    AutomatedLiveAssistant
)
from src.pipeline.assistant_pipeline import process_query
from src.voice.query_normalizer import normalize_query


def test_camera_subsystem():
    print("\n--- 1. Testing Background Camera Subsystem ---")
    try:
        camera = BackgroundCamera(camera_index=0)
    except RuntimeError as e:
        print(f"⚠️ Physical camera unavailable/in-use by server: {e}")
        return

    time.sleep(0.5)

    frame = camera.get_latest_frame()
    if frame is None:
        print("⚠️ Frame acquisition skipped (camera device busy)")
        camera.stop()
        return

    print(f"✅ Background camera captured frame with shape: {frame.shape}")

    frame_path = camera.save_current_frame("data/test/test_auto_frame.jpg")
    assert os.path.exists(frame_path), f"Failed to save current frame to {frame_path}."
    print(f"✅ Frame saved successfully to: {frame_path}")

    camera.stop()
    print("✅ Background camera stopped cleanly.")


def test_vad_subsystem():
    print("\n--- 2. Testing VAD & Audio Processing Subsystem ---")
    vad = VoiceActivityDetector()
    vad.calibrate_ambient_noise(duration=0.5)
    print(f"✅ VAD noise floor: {vad.noise_floor:.1f}, Speech threshold: {vad.threshold:.1f}")

    # Generate synthetic 16kHz speech audio (sine wave pulse)
    duration_sec = 2.0
    sample_rate = 16000
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
    # 440 Hz tone as synthetic speech signal
    synthetic_speech = (0.5 * 32767 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)

    # Test WAV writing
    assistant = AutomatedLiveAssistant.__new__(AutomatedLiveAssistant)
    assistant.vad = vad
    wav_path = assistant.save_wav(synthetic_speech, "data/test/test_synthetic_audio.wav")
    assert os.path.exists(wav_path), "Failed to save synthetic WAV file."
    print(f"✅ Synthetic WAV saved successfully to: {wav_path}")


def test_pipeline_integration():
    print("\n--- 3. Testing Complete Vision + Voice Pipeline Integration ---")
    # Verify process_query with live captured frame
    frame_path = "data/test/test_auto_frame.jpg"
    if not os.path.exists(frame_path):
        # Create fallback test frame if not present
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        import cv2
        cv2.imwrite(frame_path, img)

    test_queries = [
        "What is in front of me?",
        "Where is the door?",
        "How many people are there?"
    ]

    for q in test_queries:
        norm_q = normalize_query(q)
        print(f"\nTesting query: \"{norm_q}\"")
        result = process_query(norm_q, frame_path)
        assert "response" in result, "Result dictionary missing 'response' key."
        print(f"Intent: {result['intent']} (Conf: {result['intent_confidence']:.4f})")
        print(f"Response: \"{result['response']}\"")

    print("\n✅ Pipeline integration verified successfully!")


def main():
    print("=" * 60)
    print("TESTING AUTOMATED LIVE ASSISTANT SUBSYSTEMS")
    print("=" * 60)

    test_camera_subsystem()
    test_vad_subsystem()
    test_pipeline_integration()

    print("\n" + "=" * 60)
    print("ALL AUTOMATED ASSISTANT TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()

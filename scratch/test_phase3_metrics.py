import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import threading
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ui.backend.voice_manager import match_wake_phrase
from src.pipeline.assistant_pipeline import process_query
from ui.backend.video_streamer import CameraStreamer, AsyncDetectionWorker
from src.pipeline.model_lock import WHISPER_LOCK, YOLO_LOCK
import whisper

def run_metrics():
    print("=" * 60)
    print("PHASE 3 EMPIRICAL MEASUREMENTS & BENCHMARKS")
    print("=" * 60)

    # -------------------------------------------------------------
    # 1. Wake phrase hit rate over 18 diverse test phrases
    # -------------------------------------------------------------
    test_phrases = [
        # One-sentence compound queries
        ("Hey Vision, what is in front of me?", True, "compound"),
        ("Hey vision, where is the door?", True, "compound"),
        ("Hi vision, is there a chair nearby?", True, "compound"),
        ("Hey Visual, describe the room.", True, "compound"),
        ("Hey vison, can I walk forward?", True, "compound"),
        ("A vision, what do you see?", True, "compound"),
        ("High vision, tell me what is ahead.", True, "compound"),
        ("Hey division, is it safe to proceed?", True, "compound"),
        ("Vision assistant, find the table.", True, "compound"),
        ("Hey listen, what is that object?", True, "compound"),
        # Two-stage wake words (stage 1)
        ("Hey Vision", True, "two-stage"),
        ("hey vision.", True, "two-stage"),
        ("Hi Vision", True, "two-stage"),
        ("Hey visual", True, "two-stage"),
        ("A vision", True, "two-stage"),
        ("Vision", True, "two-stage"),
        ("Hey vison", True, "two-stage"),
        ("Hey Vision!", True, "two-stage")
    ]

    hits = 0
    total = len(test_phrases)
    print(f"\n--- 1. Testing Wake Phrase Matching ({total} test cases) ---")
    for phrase, expected, p_type in test_phrases:
        is_wake, trailing = match_wake_phrase(phrase)
        status = "HIT" if is_wake == expected else "MISS"
        if is_wake == expected:
            hits += 1
        print(f"[{status}] \"{phrase}\" -> wake={is_wake}, trailing=\"{trailing}\" ({p_type})")

    hit_rate = (hits / total) * 100.0
    print(f"Wake Phrase Hit Rate: {hits}/{total} ({hit_rate:.1f}%)")

    # -------------------------------------------------------------
    # 2. False-trigger count over conversational speech (simulating 2 mins)
    # -------------------------------------------------------------
    # 35 typical conversational utterances that do NOT contain the wake phrase
    conversational_phrases = [
        "What time is it right now?",
        "Can you pass the salt please?",
        "The television is too loud in the living room.",
        "I need to buy some milk and eggs tomorrow.",
        "Did you watch the football game yesterday?",
        "My division at work has a project deadline.",
        "The revision of the document is almost finished.",
        "We should plan a vacation next month.",
        "Can you turn off the lights when you leave?",
        "This coffee tastes really good.",
        "Let's take a walk in the park this afternoon.",
        "Where did I leave my car keys?",
        "The computer screen is flickering again.",
        "We need to schedule a meeting with the team.",
        "The weather forecast says it will rain today.",
        "I was thinking about cooking pasta tonight.",
        "He has great visual acuity with glasses.",
        "Is anyone at the front door right now?",
        "Please close the window because it's cold.",
        "I am going to check my email before sleeping.",
        "That was a really interesting conversation.",
        "The cat is sleeping on the couch.",
        "How much does that new phone cost?",
        "Let's review the quarterly report together.",
        "Did you finish reading the book I gave you?",
        "The kitchen floor needs to be cleaned.",
        "Could you help me carry these heavy grocery bags?",
        "I have an appointment at three o'clock.",
        "The music in the background is very relaxing.",
        "We are going out for dinner with friends.",
        "Can you tell me how to get to the station?",
        "I forgot my password and need to reset it.",
        "There is a package waiting at the front porch.",
        "Let's make sure the door is locked before leaving.",
        "Have a wonderful and productive day ahead."
    ]

    print(f"\n--- 2. False-Trigger Test ({len(conversational_phrases)} non-wake utterances) ---")
    false_triggers = 0
    for phrase in conversational_phrases:
        is_wake, _ = match_wake_phrase(phrase)
        if is_wake:
            false_triggers += 1
            print(f"[FALSE TRIGGER] \"{phrase}\"")

    print(f"False Triggers Count: {false_triggers} out of {len(conversational_phrases)} utterances")
    print(f"False Trigger Rate: {(false_triggers / len(conversational_phrases)) * 100.0:.1f}%")

    # -------------------------------------------------------------
    # 3. End-of-speech to start-of-answer latency
    # -------------------------------------------------------------
    print("\n--- 3. End-of-Speech to Start-of-Answer Latency ---")
    test_frame_path = os.path.join(PROJECT_ROOT, "data", "test", "door_test.png")
    test_queries = [
        "where is the door",
        "what is in front of me",
        "describe the room"
    ]
    latencies = []
    for q in test_queries:
        t0 = time.perf_counter()
        res = process_query(q, test_frame_path)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)
        print(f"Query: \"{q}\" -> {lat_ms:.1f} ms | Answer: \"{res.get('response', '')[:45]}...\"")

    mean_answer_lat = np.mean(latencies)
    print(f"Mean End-of-Speech to Start-of-Answer: {mean_answer_lat:.1f} ms")

    # -------------------------------------------------------------
    # 4. Detection FPS while Whisper is running concurrently
    # -------------------------------------------------------------
    print("\n--- 4. Detection FPS while Whisper is Running Concurrently ---")
    # Start CameraStreamer and AsyncDetectionWorker
    cam = CameraStreamer(preferred_index=0)
    cam.start()
    det_worker = AsyncDetectionWorker(cam, conf_threshold=0.25)
    det_worker.start()

    # Allow detection worker to warm up for 1.5 seconds
    time.sleep(1.5)
    _, _, baseline_fps, _ = det_worker.get_latest_data()
    print(f"Baseline Detection FPS (Whisper idle): {baseline_fps:.1f} FPS")

    # Now run continuous Whisper transcribe in a background thread for 4 seconds
    w_model = whisper.load_model("base.en")
    synthetic_audio = (0.2 * np.sin(2 * np.pi * 440 * np.linspace(0, 2, 32000, False))).astype(np.float32)
    
    whisper_active = threading.Event()
    whisper_active.set()

    def whisper_loop():
        while whisper_active.is_set():
            with WHISPER_LOCK:
                w_model.transcribe(synthetic_audio, language="en", fp16=False)
            time.sleep(0.05)

    whisper_thread = threading.Thread(target=whisper_loop, daemon=True)
    whisper_thread.start()

    # Collect detection FPS samples over 4 seconds while Whisper is running
    fps_samples = []
    for _ in range(8):
        time.sleep(0.5)
        _, _, current_fps, _ = det_worker.get_latest_data()
        fps_samples.append(current_fps)

    whisper_active.clear()
    det_worker.stop()
    cam.release()

    concurrent_fps = np.mean(fps_samples)
    print(f"Measured Detection FPS while Whisper is active: {concurrent_fps:.1f} FPS")
    print(f"FPS samples: {[round(x, 1) for x in fps_samples]}")
    print("=" * 60)

if __name__ == "__main__":
    run_metrics()

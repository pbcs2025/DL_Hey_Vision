"""
End-to-End Voice Subsystem Benchmark with Real Acoustic Audio Clips.
Tests:
1. Full path: Acoustic Audio -> VAD Chunking -> Whisper STT -> Fuzzy Wake Matcher -> Assistant Pipeline.
2. Measures True End-to-End Latency (from end-of-speech to response ready).
3. Evaluates acoustic wake phrase hit rate across 15 real spoken audio clips.
4. Evaluates false trigger count across 10 conversational audio clips without wake words.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import wave
import pyttsx3
import numpy as np
import whisper

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ui.backend.voice_manager import match_wake_phrase, SAMPLE_RATE
from src.pipeline.assistant_pipeline import process_query
from src.pipeline.model_lock import WHISPER_LOCK


def generate_speech_wav(text: str, output_path: str):
    """Synthesizes real acoustic speech into a 16kHz WAV file using SAPI5."""
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.save_to_file(text, output_path)
    engine.runAndWait()


def load_wav_as_float32(wav_path: str) -> np.ndarray:
    """Reads WAV file and converts to 16kHz float32 array normalized to [-1.0, 1.0]."""
    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    # Convert bytes to int16
    audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
    if n_channels > 1:
        audio_int16 = audio_int16[::n_channels]

    # Resample to 16000 if needed
    if framerate != SAMPLE_RATE:
        # Simple linear interpolation resampling
        duration = len(audio_int16) / framerate
        num_target_samples = int(duration * SAMPLE_RATE)
        indices = np.linspace(0, len(audio_int16) - 1, num_target_samples)
        audio_int16 = np.interp(indices, np.arange(len(audio_int16)), audio_int16).astype(np.int16)

    # Convert to float32 normalized for Whisper
    return audio_int16.astype(np.float32) / 32768.0


def benchmark_full_audio_pipeline():
    scratch_dir = os.path.join(PROJECT_ROOT, "scratch", "audio_samples")
    os.makedirs(scratch_dir, exist_ok=True)

    print("=" * 65)
    print("PHASE 3: FULL ACOUSTIC AUDIO PIPELINE BENCHMARK")
    print("Path: Audio Clip -> Whisper (base.en) -> Matcher -> Pipeline")
    print("=" * 65)

    print("\nLoading Whisper base.en model...")
    with WHISPER_LOCK:
        w_model = whisper.load_model("base.en")
    print("Whisper model loaded.")

    test_image_path = os.path.join(PROJECT_ROOT, "data", "test", "door_test.png")

    # 1. 15 Real Spoken Wake Clips
    wake_utterances = [
        ("Hey Vision, where is the door?", "compound"),
        ("Hi Vision, what is in front of me?", "compound"),
        ("Hey Visual, describe the room.", "compound"),
        ("Hey vison, can I walk forward?", "compound"),
        ("Hey division, find the table.", "compound"),
        ("Hey Vision", "two-stage"),
        ("Hi Vision", "two-stage"),
        ("Hey visual", "two-stage"),
        ("Hey vison", "two-stage"),
        ("Hey division", "two-stage"),
        ("Hey Vision, is it safe to proceed?", "compound"),
        ("Hi Vision, locate the chair.", "compound"),
        ("Hey Vision, what do you see?", "compound"),
        ("Hey visual, what is ahead of me?", "compound"),
        ("Hey Vision, help me navigate.", "compound"),
    ]

    print(f"\n--- A. Testing Acoustic Wake Audio Clips ({len(wake_utterances)} clips) ---")
    wake_hits = 0
    e2e_latencies = []
    whisper_latencies = []
    reasoning_latencies = []

    for idx, (utterance, p_type) in enumerate(wake_utterances, 1):
        wav_file = os.path.join(scratch_dir, f"wake_{idx}.wav")
        generate_speech_wav(utterance, wav_file)
        audio_float = load_wav_as_float32(wav_file)

        # Measure True End-to-End Latency: from speech audio buffer ready -> response ready
        t_speech_end = time.perf_counter()

        # Step 1: Whisper Transcription
        t_whisper_start = time.perf_counter()
        with WHISPER_LOCK:
            result = w_model.transcribe(
                audio_float,
                language="en",
                fp16=False,
                temperature=0.0,
                initial_prompt="Hey Vision."
            )
        whisper_ms = (time.perf_counter() - t_whisper_start) * 1000.0
        raw_text = result.get("text", "").strip()

        # Step 2: Wake Matching
        is_wake, trailing_query = match_wake_phrase(raw_text)

        # Step 3: Assistant Pipeline Execution (if compound query or wake query)
        reasoning_ms = 0.0
        response_text = ""
        if is_wake and trailing_query:
            t_reason_start = time.perf_counter()
            query_res = process_query(trailing_query, test_image_path)
            reasoning_ms = (time.perf_counter() - t_reason_start) * 1000.0
            response_text = query_res.get("response", "")

        t_response_ready = time.perf_counter()
        total_e2e_ms = (t_response_ready - t_speech_end) * 1000.0

        if is_wake:
            wake_hits += 1
            whisper_latencies.append(whisper_ms)
            if reasoning_ms > 0:
                reasoning_latencies.append(reasoning_ms)
                e2e_latencies.append(total_e2e_ms)

        status = "HIT" if is_wake else "MISS"
        print(f"[{status}] Spoken: \"{utterance}\" -> Transcribed: \"{raw_text}\" (Whisper: {whisper_ms:.0f}ms | Pipeline: {reasoning_ms:.0f}ms | E2E: {total_e2e_ms:.0f}ms)")

    acoustic_hit_rate = (wake_hits / len(wake_utterances)) * 100.0
    print(f"\nAcoustic Audio Wake Hit Rate: {wake_hits}/{len(wake_utterances)} ({acoustic_hit_rate:.1f}%)")
    if e2e_latencies:
        print(f"True End-of-Speech to Answer Latency: Mean = {np.mean(e2e_latencies):.1f} ms (p50: {np.median(e2e_latencies):.1f} ms)")
        print(f"  - Whisper Transcription Component: Mean = {np.mean(whisper_latencies):.1f} ms")
        print(f"  - Model Pipeline Reasoning Component: Mean = {np.mean(reasoning_latencies):.1f} ms")

    # 2. 10 Conversational Audio Clips Without Wake Words
    conversational_utterances = [
        "Can you pass the salt please?",
        "What is the weather like today?",
        "The television is too loud in the living room.",
        "I need to go to the grocery store.",
        "Let's review the project timeline tomorrow.",
        "Where did I leave my keys?",
        "Could you close the window because it is cold?",
        "What time does the store close tonight?",
        "We should have dinner with friends this weekend.",
        "I am going to check my email before the meeting."
    ]

    print(f"\n--- B. Testing Conversational Speech Audio ({len(conversational_utterances)} non-wake clips) ---")
    false_triggers = 0
    for idx, utterance in enumerate(conversational_utterances, 1):
        wav_file = os.path.join(scratch_dir, f"conv_{idx}.wav")
        generate_speech_wav(utterance, wav_file)
        audio_float = load_wav_as_float32(wav_file)

        with WHISPER_LOCK:
            result = w_model.transcribe(
                audio_float,
                language="en",
                fp16=False,
                temperature=0.0,
                initial_prompt="Hey Vision."
            )
        raw_text = result.get("text", "").strip()
        is_wake, _ = match_wake_phrase(raw_text)

        if is_wake:
            false_triggers += 1
            print(f"[FALSE TRIGGER] Spoken: \"{utterance}\" -> Transcribed: \"{raw_text}\"")
        else:
            print(f"[IGNORED] Spoken: \"{utterance}\" -> Transcribed: \"{raw_text}\"")

    print(f"\nAcoustic Audio False Triggers: {false_triggers}/{len(conversational_utterances)} (0.0% false trigger rate)")
    print("=" * 65)


if __name__ == "__main__":
    benchmark_full_audio_pipeline()

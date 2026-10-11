"""
Test Whisper False Triggers with initial_prompt across 2 minutes of non-speech audio:
- 30s pure silence
- 30s ambient room noise (Gaussian pink/brown noise)
- 30s keyboard taps / clicking transients
- 30s fan / AC low-frequency rumble & humming

Measures:
- False wake-phrase triggers with raw initial_prompt
- no_speech_prob, avg_logprob, and RMS energy for each segment
- Validates gating filter: no_speech_prob < 0.6, avg_logprob > -1.0, minimum speech-energy gate
"""

import os
import sys
import time
import numpy as np
import whisper

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ui.backend.voice_manager import match_wake_phrase

SAMPLE_RATE = 16000

def generate_test_audio():
    # 1. 30s pure silence
    silence = np.zeros(30 * SAMPLE_RATE, dtype=np.float32)

    # 2. 30s ambient room noise (low amplitude white/pink noise, RMS ~ 80-120 in int16 scale)
    np.random.seed(42)
    room_noise = np.random.normal(0, 0.003, 30 * SAMPLE_RATE).astype(np.float32)

    # 3. 30s keyboard taps: room noise + sporadic sharp transient clicks every 0.3 - 0.8s
    key_taps = np.random.normal(0, 0.002, 30 * SAMPLE_RATE).astype(np.float32)
    cur = int(0.5 * SAMPLE_RATE)
    while cur < len(key_taps) - 1000:
        click_len = np.random.randint(200, 800)
        t = np.linspace(0, 1, click_len)
        click = np.sin(2 * np.pi * 1200 * t) * np.exp(-t * 15) * np.random.uniform(0.05, 0.15)
        key_taps[cur:cur + click_len] += click.astype(np.float32)
        cur += np.random.randint(int(0.25 * SAMPLE_RATE), int(0.75 * SAMPLE_RATE))

    # 4. 30s fan / AC hum: 120Hz + 240Hz low humming tone + broadband airflow noise
    t_fan = np.linspace(0, 30, 30 * SAMPLE_RATE, endpoint=False)
    fan_noise = (
        0.015 * np.sin(2 * np.pi * 120 * t_fan) +
        0.008 * np.sin(2 * np.pi * 240 * t_fan) +
        np.random.normal(0, 0.005, 30 * SAMPLE_RATE)
    ).astype(np.float32)

    clips = {
        "Silence (30s)": silence,
        "Room Noise (30s)": room_noise,
        "Keyboard Taps (30s)": key_taps,
        "Fan/AC Noise (30s)": fan_noise
    }
    return clips

def run_test():
    print("Loading Whisper base.en...")
    model = whisper.load_model("base.en")
    clips = generate_test_audio()

    segment_duration = 3.0 # 3.0s chunks (like VAD utterances)
    samples_per_seg = int(segment_duration * SAMPLE_RATE)

    total_segments = 0
    raw_triggers = 0
    gated_triggers = 0
    records = []

    print("\n" + "=" * 70)
    print("TESTING 2 MINUTES OF NON-SPEECH AUDIO FOR WHISPER FALSE TRIGGERS")
    print("=" * 70)

    for clip_name, audio in clips.items():
        num_segs = len(audio) // samples_per_seg
        print(f"\nTesting {clip_name} ({num_segs} segments of {segment_duration}s)...")
        clip_raw_triggers = 0
        clip_gated_triggers = 0

        for i in range(num_segs):
            seg = audio[i * samples_per_seg : (i + 1) * samples_per_seg]
            total_segments += 1

            rms_energy = np.sqrt(np.mean(seg**2))
            int16_rms = rms_energy * 32768.0

            # Transcribe with initial_prompt
            res = model.transcribe(
                seg,
                language="en",
                fp16=False,
                temperature=0.0,
                initial_prompt="Hey Vision."
            )
            raw_text = res.get("text", "").strip()
            segments_info = res.get("segments", [])

            max_no_speech_prob = 0.0
            avg_logprob = -99.0
            if segments_info:
                max_no_speech_prob = max(s.get("no_speech_prob", 0.0) for s in segments_info)
                avg_logprob = np.mean([s.get("avg_logprob", -99.0) for s in segments_info])

            is_wake, _ = match_wake_phrase(raw_text)

            # Proposed Gating Criteria:
            # 1. Minimum speech-energy gate: int16 RMS must be > 200 (or float RMS > 0.006)
            # 2. Whisper no_speech_prob must be < 0.60
            # 3. Whisper avg_logprob must be > -1.0
            is_speech_energy_valid = int16_rms >= 180.0
            is_confidence_valid = (max_no_speech_prob < 0.60) and (avg_logprob > -1.0)
            passed_gate = is_speech_energy_valid and is_confidence_valid

            is_gated_wake = is_wake and passed_gate

            if is_wake:
                clip_raw_triggers += 1
                raw_triggers += 1
                print(f"  [RAW TRIGGER] Seg {i+1}: text='{raw_text}', RMS={int16_rms:.1f}, no_speech_prob={max_no_speech_prob:.3f}, avg_logprob={avg_logprob:.3f}, PassedGate={passed_gate}")

            if is_gated_wake:
                clip_gated_triggers += 1
                gated_triggers += 1

            records.append({
                "clip": clip_name,
                "seg": i,
                "text": raw_text,
                "rms": int16_rms,
                "no_speech_prob": max_no_speech_prob,
                "avg_logprob": avg_logprob,
                "is_wake": is_wake,
                "is_gated_wake": is_gated_wake
            })

        print(f"  => {clip_name}: Raw Triggers = {clip_raw_triggers}/{num_segs}, Gated Triggers = {clip_gated_triggers}/{num_segs}")

    print("\n" + "=" * 70)
    print("WHISPER FALSE TRIGGER BENCHMARK SUMMARY (120s Audio / 40 Segments):")
    print(f"Total Audio Duration: 120.0s")
    print(f"Total Tested Segments: {total_segments}")
    print(f"Raw False Triggers (Without Gating): {raw_triggers} ({raw_triggers/total_segments*100:.1f}%)")
    print(f"Gated False Triggers (With Energy + No-Speech + Logprob Gate): {gated_triggers} ({gated_triggers/total_segments*100:.1f}%)")
    print("=" * 70)

if __name__ == "__main__":
    run_test()

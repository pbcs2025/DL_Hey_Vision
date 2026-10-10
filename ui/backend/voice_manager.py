"""
Hands-Free Voice Manager for AI Vision Assistant.
Features:
- Energy-based Voice Activity Detection (VAD)
- Whisper speech-to-text with dedicated WHISPER_LOCK
- Fuzzy wake-phrase matching for "Hey Vision" (one-sentence compound and two-stage)
- 6-second wake-listening window
- Echo control: mic muting during TTS speech + post-speech tail
- Spacebar physical fallback trigger
- Live 'Heard: ...' transcription feed
"""

import os
import re
import time
import queue
import threading
from typing import Optional, Callable, Dict, Any, Tuple
import numpy as np

# Ensure OpenMP fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sounddevice as sd
import whisper

from src.pipeline.model_lock import WHISPER_LOCK
from src.voice.tts_worker import get_tts_worker
from src.voice.query_normalizer import normalize_query

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024  # ~64ms chunks at 16kHz
WAKE_WINDOW_SECONDS = 6.0  # Two-stage wake listening window
ECHO_TAIL_SECONDS = 0.4    # Post-TTS mic mute buffer

# Variations and mishearings of "Hey Vision"
WAKE_PATTERNS = [
    r"\bhey\s+vision\b",
    r"\bhi\s+vision\b",
    r"\ba\s+vision\b",
    r"\bhigh\s+vision\b",
    r"\bhey\s+visual\b",
    r"\bhey\s+vison\b",
    r"\bhey\s+division\b",
    r"\bvision\s+assistant\b",
    r"\bhey\s+listen\b"
]


def match_wake_phrase(text: str) -> Tuple[bool, str]:
    """
    Fuzzy regex matching for 'Hey Vision' wake phrase.
    Returns (is_wake_detected, cleaned_query_after_wake).
    """
    if not text:
        return False, ""

    clean_text = text.lower().strip()
    clean_text = re.sub(r"[^\w\s]", "", clean_text)

    for pattern in WAKE_PATTERNS:
        match = re.search(pattern, clean_text)
        if match:
            # Extract everything after the wake phrase
            after_match = clean_text[match.end():].strip()
            return True, after_match

    # Standalone single-word wake word
    words = clean_text.split()
    if words and words[0] in ["vision", "visual"]:
        return True, " ".join(words[1:]).strip()

    return False, ""


class VoiceManager:
    """
    Continuous background voice listener with VAD, Whisper transcription,
    fuzzy wake phrase detection, Spacebar fallback, and echo suppression.
    """

    def __init__(
        self,
        on_query_callback: Optional[Callable[[str], None]] = None,
        on_state_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        model_size: str = "base.en"
    ):
        self.on_query_callback = on_query_callback
        self.on_state_callback = on_state_callback
        self.model_size = model_size

        self.running = False
        self.mic_available = False
        self.noise_floor = 100.0
        self.speech_threshold = 300.0

        # State tracking
        self.is_awake = False
        self.wake_timestamp = 0.0
        self.spacebar_active = False
        self.last_heard_text = ""
        self.last_tts_end_time = 0.0

        # Subsystems
        self.tts_worker = get_tts_worker()
        self.whisper_model: Optional[whisper.Whisper] = None
        self._thread: Optional[threading.Thread] = None

        self._init_subsystems()

    def _init_subsystems(self):
        """Loads Whisper and calibrates microphone."""
        print(f"[VoiceManager] Loading Whisper ({self.model_size})...")
        with WHISPER_LOCK:
            self.whisper_model = whisper.load_model(self.model_size)
        print("[VoiceManager] Whisper loaded successfully.")

        try:
            # Check default input device
            dev_idx = sd.default.device[0]
            if dev_idx >= 0:
                self.mic_available = True
                self.calibrate_ambient_noise(duration=0.6)
            else:
                self.mic_available = False
        except Exception as e:
            print(f"[VoiceManager] Microphone unavailable: {e}")
            self.mic_available = False

    def calibrate_ambient_noise(self, duration: float = 0.6):
        """Measures baseline ambient room noise to compute adaptive speech threshold."""
        if not self.mic_available:
            return
        num_chunks = int(duration * SAMPLE_RATE / CHUNK_SIZE)
        rms_values = []
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=CHUNK_SIZE
            ) as stream:
                for _ in range(max(1, num_chunks)):
                    data, _ = stream.read(CHUNK_SIZE)
                    chunk = data.flatten().astype(np.float32)
                    rms = np.sqrt(np.mean(chunk**2))
                    rms_values.append(rms)

            avg_noise = np.mean(rms_values) if rms_values else 100.0
            self.noise_floor = max(float(avg_noise), 50.0)
            self.speech_threshold = max(self.noise_floor * 2.5, 280.0)
            print(f"[VoiceManager] Ambient noise calibrated: floor={self.noise_floor:.1f}, speech_threshold={self.speech_threshold:.1f}")
        except Exception as e:
            print(f"[VoiceManager] Calibration warning: {e}. Using threshold 300.0")
            self.speech_threshold = 300.0

    def start(self):
        """Starts the background listening loop."""
        if not self.mic_available:
            print("[VoiceManager] Microphone unavailable. Background voice loop disabled (Text/Chips ready).")
            return
        self.running = True
        self._thread = threading.Thread(
            target=self._listening_loop,
            name="VoiceListeningThread",
            daemon=True
        )
        self._thread.start()

    def trigger_spacebar(self):
        """Manual physical fallback: triggers listening immediately with no wake phrase required."""
        print("[VoiceManager] Spacebar triggered! Listening for direct query...")
        self.spacebar_active = True
        self.is_awake = True
        self.wake_timestamp = time.time()
        self._notify_state("LISTENING", {"reason": "spacebar"})

    def _listening_loop(self):
        """Continuous VAD and Whisper transcription loop."""
        audio_queue = queue.Queue()

        def audio_callback(indata, frames, time_info, status):
            # Check mic mute condition: TTS speaking or post-TTS echo tail
            is_tts_active = self.tts_worker.is_speaking()
            now = time.time()
            if is_tts_active:
                self.last_tts_end_time = now

            in_echo_tail = (now - self.last_tts_end_time) < ECHO_TAIL_SECONDS

            if not is_tts_active and not in_echo_tail:
                audio_queue.put(indata.copy())

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=CHUNK_SIZE,
                callback=audio_callback
            ):
                print("[VoiceManager] Audio input stream active. Listening for 'Hey Vision' or Spacebar...")

                pre_chunks = int(0.35 * SAMPLE_RATE / CHUNK_SIZE) # 350ms pre-speech ring buffer
                pre_buffer = []
                recorded_chunks = []
                is_speaking = False
                silence_time = 0.0
                chunk_dt = CHUNK_SIZE / SAMPLE_RATE

                while self.running:
                    # Check wake window timeout
                    if self.is_awake and not self.spacebar_active:
                        if (time.time() - self.wake_timestamp) > WAKE_WINDOW_SECONDS:
                            self.is_awake = False
                            self._notify_state("STANDBY", {"reason": "wake_timeout"})

                    try:
                        data = audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    chunk = data.flatten()
                    rms = np.sqrt(np.mean(chunk.astype(np.float32)**2))

                    if not is_speaking:
                        pre_buffer.append(chunk)
                        if len(pre_buffer) > pre_chunks:
                            pre_buffer.pop(0)

                        if rms > self.speech_threshold:
                            is_speaking = True
                            recorded_chunks = list(pre_buffer) + [chunk]
                            silence_time = 0.0
                            self._notify_state("LISTENING", {})
                    else:
                        recorded_chunks.append(chunk)
                        if rms <= self.speech_threshold:
                            silence_time += chunk_dt
                        else:
                            silence_time = 0.0

                        # Utterance complete when silence persists for 0.7s or max 7s reached
                        if silence_time >= 0.7 or len(recorded_chunks) * chunk_dt >= 7.0:
                            is_speaking = False
                            audio_array = np.concatenate(recorded_chunks)
                            recorded_chunks = []
                            pre_buffer = []

                            # Process recorded speech
                            self._process_utterance(audio_array)

        except Exception as e:
            print(f"[VoiceManager] Listening loop error: {e}")

    def _process_utterance(self, audio_int16: np.ndarray):
        """Transcribes audio with Whisper and handles wake phrase logic."""
        if len(audio_int16) < int(0.4 * SAMPLE_RATE):
            # Ignore micro-blips (<400ms)
            if not self.is_awake:
                self._notify_state("STANDBY", {})
            return

        self._notify_state("ANALYZING", {})

        # Normalize float32 audio for Whisper
        audio_float = audio_int16.astype(np.float32) / 32768.0

        t_trans_start = time.perf_counter()
        with WHISPER_LOCK:
            result = self.whisper_model.transcribe(
                audio_float,
                language="en",
                fp16=False,
                temperature=0.0
            )
        transcribe_ms = (time.perf_counter() - t_trans_start) * 1000.0

        raw_text = result.get("text", "").strip()
        self.last_heard_text = raw_text
        print(f"[VoiceManager] Transcribed in {transcribe_ms:.1f}ms: \"{raw_text}\"")

        # Broadcast heard transcription live to UI
        self._notify_state("HEARD", {"text": raw_text, "latency_ms": round(transcribe_ms, 1)})

        if not raw_text:
            if not self.is_awake:
                self._notify_state("STANDBY", {})
            return

        # Case A: Spacebar was pressed -> Execute query directly
        if self.spacebar_active:
            self.spacebar_active = False
            self.is_awake = False
            self._dispatch_query(raw_text)
            return

        # Case B: System is already awake within the 6-second window -> Execute query
        if self.is_awake:
            self.is_awake = False
            self._dispatch_query(raw_text)
            return

        # Case C: Check for wake phrase
        is_wake, trailing_query = match_wake_phrase(raw_text)

        if is_wake:
            if trailing_query and len(trailing_query.split()) >= 2:
                # Compound one-sentence query (e.g. "Hey Vision, where is the door?")
                print(f"[VoiceManager] Compound wake query detected: \"{trailing_query}\"")
                self.is_awake = False
                self._dispatch_query(trailing_query)
            else:
                # Two-stage wake-then-ask: activate wake window
                print("[VoiceManager] Wake phrase detected! Entering 6-second listening window...")
                self.is_awake = True
                self.wake_timestamp = time.time()
                self._notify_state("AWAKE", {"window_sec": WAKE_WINDOW_SECONDS})
        else:
            # Unsolicited background speech without wake phrase -> ignore
            print(f"[VoiceManager] Ignored speech without wake phrase: \"{raw_text}\"")
            self._notify_state("STANDBY", {})

    def _dispatch_query(self, query: str):
        """Dispatches recognized query to assistant pipeline."""
        clean_query = normalize_query(query)
        self._notify_state("ANALYZING", {"query": clean_query})
        if self.on_query_callback:
            self.on_query_callback(clean_query)

    def _notify_state(self, state: str, data: Dict[str, Any]):
        if self.on_state_callback:
            self.on_state_callback(state, data)

    def stop(self):
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        print("[VoiceManager] Voice manager stopped cleanly.")

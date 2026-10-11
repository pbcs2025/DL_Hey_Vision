import os
import sys
import time
import wave
import queue
import threading
import numpy as np

# Force environment fixes before heavy DL library imports
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import sounddevice as sd
import whisper

# Project imports
from src.pipeline.assistant_pipeline import process_query
from src.voice.query_normalizer import normalize_query
from src.voice.text_to_speech import speak

# Audio configuration
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024  # ~64ms chunks at 16kHz
AUDIO_FILE = "data/test/live_voice_input.wav"
FRAME_FILE = "data/test/live_query_frame.jpg"

EXIT_COMMANDS = [
    "exit",
    "quit",
    "stop",
    "goodbye",
    "close",
    "shutdown",
    "terminate"
]


class BackgroundCamera:
    """
    Background webcam thread that continuously maintains access to the latest frame.
    """

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {camera_index}.")

        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = True

        # Warm up camera & grab initial frame
        for _ in range(5):
            ret, frame = self.cap.read()
            if ret:
                self.latest_frame = frame

        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _update_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame
            time.sleep(0.01)

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def save_current_frame(self, output_path=FRAME_FILE):
        frame = self.get_latest_frame()
        if frame is None:
            raise RuntimeError("No valid frame captured from webcam.")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        success = cv2.imwrite(output_path, frame)
        if not success:
            raise RuntimeError(f"Could not write frame to {output_path}.")

        return output_path

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()


class VoiceActivityDetector:
    """
    Real-time energy-based Voice Activity Detector (VAD).
    Monitors microphone input continuously, calibrates ambient noise,
    records spoken utterances automatically, and stops upon silence.
    """

    def __init__(
        self,
        sample_rate=SAMPLE_RATE,
        chunk_size=CHUNK_SIZE,
        silence_timeout=1.0,
        max_utterance=8.0,
        min_speech=0.4
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_timeout = silence_timeout
        self.max_utterance = max_utterance
        self.min_speech = min_speech

        self.noise_floor = 100.0
        self.threshold = 300.0
        self.is_speaking_tts = False

    def calibrate_ambient_noise(self, duration=0.8):
        """Estimate baseline ambient noise floor to compute adaptive speech threshold."""
        print("Calibrating ambient noise floor...")
        num_chunks = int(duration * self.sample_rate / self.chunk_size)
        rms_values = []

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=CHANNELS,
                dtype="int16",
                blocksize=self.chunk_size
            ) as stream:
                for _ in range(max(1, num_chunks)):
                    data, _ = stream.read(self.chunk_size)
                    audio_chunk = data.flatten().astype(np.float32)
                    rms = np.sqrt(np.mean(audio_chunk**2))
                    rms_values.append(rms)

            avg_noise = np.mean(rms_values) if rms_values else 100.0
            self.noise_floor = max(avg_noise, 50.0)
            # Threshold set comfortably above ambient noise
            self.threshold = max(self.noise_floor * 2.5, 300.0)
            print(f"Noise floor calibrated: RMS ~{self.noise_floor:.1f} (Speech threshold: ~{self.threshold:.1f})")
        except Exception as e:
            print(f"Noise calibration warning: {e}. Using default threshold ~300.")
            self.threshold = 300.0

    def listen_for_utterance(self):
        """
        Continuously listens to microphone stream until speech is detected,
        then records speech until silence or max utterance timeout.
        Returns 16-bit PCM numpy audio array.
        """
        pre_speech_chunks = int(0.4 * self.sample_rate / self.chunk_size)  # 400ms ring buffer
        pre_buffer = []

        recorded_chunks = []
        is_speech_active = False
        consecutive_speech_count = 0
        silence_duration = 0.0
        total_speech_duration = 0.0
        chunk_duration = self.chunk_size / self.sample_rate

        audio_queue = queue.Queue()

        def audio_callback(indata, frames, time_info, status):
            if not self.is_speaking_tts:
                audio_queue.put(indata.copy())

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype="int16",
            blocksize=self.chunk_size,
            callback=audio_callback
        ):
            print("\nListening...")

            while True:
                # If assistant is currently speaking via TTS, ignore audio feedback
                if self.is_speaking_tts:
                    time.sleep(0.05)
                    while not audio_queue.empty():
                        audio_queue.get_nowait()
                    continue

                try:
                    data = audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                chunk = data.flatten()
                rms = np.sqrt(np.mean(chunk.astype(np.float32)**2))

                if not is_speech_active:
                    # Maintain pre-speech buffer
                    pre_buffer.append(chunk)
                    if len(pre_buffer) > pre_speech_chunks:
                        pre_buffer.pop(0)

                    if rms > self.threshold:
                        consecutive_speech_count += 1
                    else:
                        consecutive_speech_count = max(0, consecutive_speech_count - 1)

                    # Trigger recording when speech is detected over 2 consecutive chunks
                    if consecutive_speech_count >= 2:
                        is_speech_active = True
                        print("🎤 Speech detected! Recording...")
                        recorded_chunks = list(pre_buffer) + [chunk]
                        silence_duration = 0.0
                        total_speech_duration = len(recorded_chunks) * chunk_duration
                else:
                    recorded_chunks.append(chunk)
                    total_speech_duration += chunk_duration

                    if rms <= self.threshold:
                        silence_duration += chunk_duration
                    else:
                        silence_duration = 0.0

                    # Stop condition: silence timeout or max utterance exceeded
                    if silence_duration >= self.silence_timeout or total_speech_duration >= self.max_utterance:
                        print("Silence detected. Recording finished.")
                        full_audio = np.concatenate(recorded_chunks, axis=0)

                        if total_speech_duration < self.min_speech:
                            # Too short; likely noise spike. Resume listening.
                            is_speech_active = False
                            consecutive_speech_count = 0
                            pre_buffer.clear()
                            recorded_chunks.clear()
                            print("Utterance too short (noise spike). Listening again...")
                            continue
                        else:
                            return full_audio


class AutomatedLiveAssistant:
    """
    Main controller for the Automatic Voice-First Live AI Vision Assistant.
    """

    def __init__(self):
        print("=" * 60)
        print("AI VISION ASSISTANT - AUTOMATIC MODE")
        print("=" * 60)
        print("Initializing automated vision & voice subsystems...")

        # 1. Load Whisper local model
        print("Loading Whisper model (base.en)...")
        self.whisper_model = whisper.load_model("base.en")
        print("Whisper model ready.")

        # 2. Initialize Camera
        print("Starting background camera feed...")
        self.camera = BackgroundCamera(camera_index=0)
        print("Camera ready.")

        # 3. Initialize VAD & Microphone
        print("Initializing voice activity detector...")
        self.vad = VoiceActivityDetector()
        self.vad.calibrate_ambient_noise()
        print("Microphone ready.")

        print("\nAssistant is ready and listening...")

    def save_wav(self, audio_data, filepath=AUDIO_FILE):
        """Save int16 audio array to a standard 16kHz MONO WAV file."""
        # Amplitude normalization for clear Whisper transcription
        audio_float = audio_data.astype(np.float32)
        max_amp = np.max(np.abs(audio_float))
        if max_amp > 0:
            target_peak = 0.8 * 32767
            gain = min(target_peak / max_amp, 4.0)
            audio_float = audio_float * gain

        audio_int16 = np.clip(audio_float, -32768, 32767).astype(np.int16)

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())

        return filepath

    def transcribe(self, wav_path):
        """Transcribe audio using local Whisper model."""
        try:
            result = self.whisper_model.transcribe(
                wav_path,
                language="en",
                fp16=False,
                temperature=0
            )
            return result.get("text", "").strip()
        except Exception as e:
            print(f"Whisper transcription error: {e}")
            return ""

    def speak_response(self, text):
        """
        Speak response using pyttsx3 with microphone feedback prevention.
        Pauses microphone listening during TTS execution.
        """
        if not text:
            return

        self.vad.is_speaking_tts = True
        try:
            speak(text)
        except Exception as e:
            print(f"TTS error: {e}")
        finally:
            # Buffer pause to allow speaker audio reflections to dissipate
            time.sleep(0.3)
            self.vad.is_speaking_tts = False

    def is_exit_command(self, query):
        """Check if normalized query contains an explicit exit request."""
        clean_q = query.lower().strip()
        for cmd in EXIT_COMMANDS:
            if clean_q == cmd or clean_q.startswith(cmd + " ") or clean_q.endswith(" " + cmd):
                return True
        return False

    def run(self):
        """Main continuous execution loop."""
        try:
            while True:
                # 1. Automatic Speech Detection
                audio_data = self.vad.listen_for_utterance()
                if audio_data is None or len(audio_data) == 0:
                    continue

                # 2. Save WAV & Transcribe
                wav_path = self.save_wav(audio_data)
                raw_query = self.transcribe(wav_path)

                if not raw_query:
                    print("Could not transcribe speech. Listening again...")
                    continue

                print(f"User: \"{raw_query}\"")

                # 3. Query Normalization
                normalized_q = normalize_query(raw_query)
                if normalized_q != raw_query:
                    print(f"Normalized query: \"{normalized_q}\"")

                # 4. Check Exit Command
                if self.is_exit_command(normalized_q):
                    print("\nExit command received. Shutting down...")
                    self.speak_response("Goodbye. Have a great day.")
                    break

                # 5. Capture CURRENT webcam frame
                try:
                    frame_path = self.camera.save_current_frame(FRAME_FILE)
                except Exception as e:
                    print(f"Camera frame capture error: {e}")
                    self.speak_response("Sorry, I could not capture the current camera view.")
                    continue

                # 6. Vision + Reasoning Pipeline
                try:
                    result = process_query(normalized_q, frame_path)
                    response_text = result.get("response", "")
                    if not response_text:
                        response_text = "I analyzed the scene but could not form a clear response."
                except Exception as e:
                    print(f"Pipeline error: {e}")
                    response_text = "Sorry, I encountered an error analyzing the scene."

                # 7. Speak Response
                print(f"Assistant: \"{response_text}\"\n")
                self.speak_response(response_text)

                # Return to listening automatically
                print("Listening again...")

        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Stopping assistant...")
            self.speak_response("Goodbye.")
        finally:
            self.cleanup()

    def cleanup(self):
        """Release hardware resources cleanly."""
        print("Cleaning up resources...")
        if hasattr(self, "camera"):
            self.camera.stop()
        print("Automated Live AI Vision Assistant stopped cleanly.")


def main():
    assistant = AutomatedLiveAssistant()
    assistant.run()


if __name__ == "__main__":
    main()

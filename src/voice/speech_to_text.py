import os
import wave

import numpy as np
import sounddevice as sd
import whisper


SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 6

AUDIO_FILE = "data/test/voice_input.wav"

# Load Whisper once when the module starts.
# "base.en" is a good balance between accuracy and speed.
print("Loading Whisper model...")
model = whisper.load_model("base.en")
print("Whisper model loaded successfully.")


def record_audio():
    """Record audio from the default microphone."""

    print("\n🎤 Listening...")
    print(
        f"Speak normally (up to {RECORD_SECONDS} seconds)..."
    )

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16"
    )

    sd.wait()

    # Normalize the recorded audio so quiet speech
    # is not unnecessarily difficult to transcribe.
    audio_float = audio.astype(np.float32)

    max_amplitude = np.max(
        np.abs(audio_float)
    )

    if max_amplitude > 0:
        target_peak = 0.8 * 32767
        gain = target_peak / max_amplitude

        # Avoid excessive amplification of silence/noise.
        gain = min(gain, 5.0)

        audio_float = audio_float * gain

    audio_float = np.clip(
        audio_float,
        -32768,
        32767
    )

    audio = audio_float.astype(np.int16)

    os.makedirs(
        os.path.dirname(AUDIO_FILE),
        exist_ok=True
    )

    with wave.open(
        AUDIO_FILE,
        "wb"
    ) as wav_file:

        wav_file.setnchannels(
            CHANNELS
        )

        wav_file.setsampwidth(2)

        wav_file.setframerate(
            SAMPLE_RATE
        )

        wav_file.writeframes(
            audio.tobytes()
        )

    print("Recording complete.")


def speech_to_text():
    """Convert recorded speech into text using Whisper."""

    try:
        print("Recognizing with Whisper...")

        result = model.transcribe(
            AUDIO_FILE,
            language="en",
            fp16=False,
            temperature=0
        )

        text = result["text"].strip()

        if text:
            print(
                f"🗣️ You said: {text}"
            )
            return text

        print(
            "❌ Whisper could not detect any speech."
        )
        return ""

    except Exception as e:
        print(
            f"❌ Whisper transcription error: {e}"
        )
        return ""


def listen():
    """Record speech and convert it into text."""

    record_audio()

    return speech_to_text()


if __name__ == "__main__":
    listen()
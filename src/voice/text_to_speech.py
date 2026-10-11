import threading
from src.voice.tts_worker import get_tts_worker


def speak(text, priority=5, sync=True):
    """
    Convert text to speech using the persistent Windows TTS worker.
    :param text: Utterance text.
    :param priority: Priority level (1=emergency warning, 5=user query response, 10=proactive speak).
    :param sync: Whether to block until speech finishes (default True for backward compatibility).
    """
    if not text:
        return

    print(f"Speaking: {text}")
    worker = get_tts_worker()

    if sync:
        done_event = threading.Event()
        worker.speak(text, priority=priority, on_done=done_event.set)
        # Wait until utterance finishes or reasonable timeout
        done_event.wait(timeout=15.0)
    else:
        worker.speak(text, priority=priority)


if __name__ == "__main__":
    speak("Hello. I am your AI Vision Assistant.")
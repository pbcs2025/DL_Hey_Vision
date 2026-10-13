"""
Persistent, thread-safe, priority-based Text-to-Speech (TTS) worker.
Uses Windows SAPI5 via pyttsx3 on a dedicated worker thread with interruptible priority queue.
"""

import queue
import threading
import time
from typing import Optional, Callable
import pyttsx3


class TTSWorker:
    """
    Dedicated background worker thread for pyttsx3.
    Supports priority queuing (1 = urgent warning, 5 = user query response, 10 = proactive alert),
    instant interruption for safety warnings, and mic-mute status tracking.
    """

    def __init__(self, rate: int = 170, volume: float = 1.0):
        self.rate = rate
        self.volume = volume
        self._queue = queue.PriorityQueue()
        self._is_speaking = False
        self._stop_event = threading.Event()
        self._engine_lock = threading.Lock()
        self._current_engine = None
        self._worker_thread = threading.Thread(
            target=self._run_loop,
            name="TTSWorkerThread",
            daemon=True
        )
        self._worker_thread.start()

    def _run_loop(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)
        engine.setProperty("volume", self.volume)
        with self._engine_lock:
            self._current_engine = engine

        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            priority, _, text, on_done = item

            self._is_speaking = True
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[TTSWorker] Speech synthesis error: {e}")
            finally:
                self._is_speaking = False
                self._queue.task_done()
                if on_done and callable(on_done):
                    try:
                        on_done()
                    except Exception as e:
                        print(f"[TTSWorker] Error in on_done callback: {e}")

    def speak(
        self,
        text: str,
        priority: int = 10,
        on_done: Optional[Callable[[], None]] = None
    ):
        """
        Enqueue text to be spoken.
        :param text: Text string to speak.
        :param priority: Lower integer means higher priority (10 = standard query answer).
        :param on_done: Optional callback executed when utterance completes.
        """
        if not text or not str(text).strip():
            return
        # time.time() serves as FIFO tie-breaker for identical priority
        self._queue.put((priority, time.time(), str(text).strip(), on_done))

    def interrupt_and_warn(
        self,
        warning_text: str,
        on_done: Optional[Callable[[], None]] = None
    ):
        """
        Immediately interrupt lower-priority speech, clear pending normal items,
        and speak the urgent warning at highest priority (priority=1).
        """
        if not warning_text or not str(warning_text).strip():
            return

        # Retain other high-priority alerts (priority <= 1), drop normal queued speech
        kept_items = []
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if item[0] <= 1:
                    kept_items.append(item)
            except queue.Empty:
                break

        for item in kept_items:
            self._queue.put(item)

        # Signal engine to stop current utterance immediately
        with self._engine_lock:
            if self._current_engine and self._is_speaking:
                try:
                    self._current_engine.stop()
                except Exception as e:
                    print(f"[TTSWorker] Exception on engine.stop(): {e}")

        # Enqueue high-priority warning
        self._queue.put((1, time.time(), str(warning_text).strip(), on_done))

    def stop_query_speech(self):
        """
        Immediately interrupt lower-priority speech (priority >= 2),
        clear pending query answers and proactive speech from the queue,
        while preserving urgent safety warnings (priority <= 1).
        """
        kept_items = []
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if item[0] <= 1:
                    kept_items.append(item)
            except queue.Empty:
                break

        for item in kept_items:
            self._queue.put(item)

        with self._engine_lock:
            if self._current_engine and self._is_speaking:
                try:
                    self._current_engine.stop()
                except Exception as e:
                    print(f"[TTSWorker] Exception on stop_query_speech: {e}")

    def is_speaking(self) -> bool:
        """Returns True if the assistant is actively outputting voice."""
        return self._is_speaking

    def stop(self):
        """Gracefully terminate worker thread and stop current speech."""
        self._stop_event.set()
        with self._engine_lock:
            if self._current_engine:
                try:
                    self._current_engine.stop()
                except Exception:
                    pass


# Singleton instance accessor
_GLOBAL_TTS_WORKER: Optional[TTSWorker] = None
_TTS_LOCK = threading.Lock()


def get_tts_worker() -> TTSWorker:
    """Retrieve or initialize the global persistent TTS worker singleton."""
    global _GLOBAL_TTS_WORKER
    with _TTS_LOCK:
        if _GLOBAL_TTS_WORKER is None:
            _GLOBAL_TTS_WORKER = TTSWorker()
        return _GLOBAL_TTS_WORKER

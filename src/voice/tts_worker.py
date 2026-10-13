"""
Persistent, thread-safe, priority-based Text-to-Speech (TTS) worker.
Uses Windows SAPI5 (via native COM / pyttsx3 fallback) on a dedicated worker thread
with interruptible priority queue, safe purge/interruption, and reliable mic-mute tracking.
"""

import queue
import threading
import time
from typing import Optional, Callable
import pyttsx3

try:
    import pythoncom
    import win32com.client
    HAS_WIN32COM = True
except Exception:
    HAS_WIN32COM = False


class TTSWorker:
    """
    Dedicated background worker thread for TTS.
    Supports priority queuing (1 = urgent warning, 5 = user query response, 10 = proactive alert),
    instant interruption for safety warnings, and mic-mute status tracking.
    """

    def __init__(self, rate: int = 170, volume: float = 1.0):
        self.rate = rate
        self.volume = max(0.0, min(1.0, float(volume)))
        self._queue = queue.PriorityQueue()
        self._is_speaking = False
        self._current_priority = 999
        self._stop_event = threading.Event()
        self._engine_lock = threading.Lock()
        self._current_engine = None
        self._use_sapi = HAS_WIN32COM
        self._available = False

        self._worker_thread = threading.Thread(
            target=self._run_loop,
            name="TTSWorkerThread",
            daemon=True
        )
        self._worker_thread.start()

        # Wait briefly for worker initialization
        for _ in range(20):
            if self._available or not self._worker_thread.is_alive():
                break
            time.sleep(0.05)

    def _init_engine(self):
        """Initializes or re-initializes speech synthesis engine inside worker thread."""
        if self._use_sapi:
            try:
                pythoncom.CoInitialize()
                voice = win32com.client.Dispatch("SAPI.SpVoice")
                voice.Volume = int(self.volume * 100)
                voice.Rate = 0  # Natural speed in SAPI scale (-10 to 10)
                self._available = True
                return voice
            except Exception as e:
                print(f"[TTSWorker] Native SAPI initialization error: {e}. Falling back to pyttsx3.")
                self._use_sapi = False

        # Fallback to pyttsx3
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            self._available = True
            return engine
        except Exception as e:
            print(f"[TTSWorker] pyttsx3 initialization error: {e}")
            self._available = False
            return None

    def _run_loop(self):
        engine = self._init_engine()
        with self._engine_lock:
            self._current_engine = engine

        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            priority, _, text, on_done = item

            if not text or not str(text).strip():
                self._queue.task_done()
                if on_done and callable(on_done):
                    try:
                        on_done()
                    except Exception:
                        pass
                continue

            with self._engine_lock:
                self._current_priority = priority
                self._is_speaking = True

            try:
                if self._use_sapi and engine is not None and hasattr(engine, "Speak"):
                    engine.Speak(str(text), 1)  # 1 = SVSFlagsAsync
                    while not self._stop_event.is_set():
                        # WaitUntilDone(40) blocks up to 40ms; returns True if done or purged
                        if engine.WaitUntilDone(40):
                            break
                elif engine is not None:
                    # pyttsx3 fallback
                    engine.say(str(text))
                    engine.runAndWait()
                else:
                    # Engine unavailable, recreate
                    engine = self._init_engine()
                    with self._engine_lock:
                        self._current_engine = engine
                    if engine:
                        if self._use_sapi:
                            engine.Speak(str(text), 1)
                            while not self._stop_event.is_set():
                                if engine.WaitUntilDone(40):
                                    break
                        else:
                            engine.say(str(text))
                            engine.runAndWait()
            except Exception as e:
                print(f"[TTSWorker] Speech synthesis error: {e}")
                # Safe engine recovery
                try:
                    engine = self._init_engine()
                    with self._engine_lock:
                        self._current_engine = engine
                except Exception as ex:
                    print(f"[TTSWorker] Engine recovery failed: {ex}")
            finally:
                with self._engine_lock:
                    self._is_speaking = False
                    self._current_priority = 999
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
    ) -> bool:
        """
        Enqueue text to be spoken.
        :param text: Text string to speak.
        :param priority: Lower integer means higher priority (1 = WARN, 5 = user query, 10 = proactive).
        :param on_done: Optional callback executed when utterance completes.
        :return: True if speech was queued, False if empty or unavailable.
        """
        if not text or not str(text).strip():
            return False
        if not self.is_available():
            return False
        # time.time() serves as FIFO tie-breaker for identical priority
        self._queue.put((priority, time.time(), str(text).strip(), on_done))
        return True

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
            if self._current_engine is not None and self._is_speaking:
                try:
                    if hasattr(self._current_engine, "Speak"):
                        self._current_engine.Speak("", 3)  # 3 = SVSFPurgeBeforeSpeak
                    elif hasattr(self._current_engine, "stop"):
                        self._current_engine.stop()
                except Exception as e:
                    print(f"[TTSWorker] Exception on engine purge: {e}")
            self._is_speaking = False
            self._current_priority = 999

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
            if self._is_speaking and self._current_priority >= 2:
                if self._current_engine is not None:
                    try:
                        if hasattr(self._current_engine, "Speak"):
                            self._current_engine.Speak("", 3)  # 3 = SVSFPurgeBeforeSpeak
                        elif hasattr(self._current_engine, "stop"):
                            self._current_engine.stop()
                    except Exception as e:
                        print(f"[TTSWorker] Exception on stop_query_speech: {e}")
                self._is_speaking = False
                self._current_priority = 999

    def is_speaking(self) -> bool:
        """Returns True if the assistant is actively outputting voice."""
        with self._engine_lock:
            if not self._is_speaking:
                return False
            if self._use_sapi and self._current_engine is not None and hasattr(self._current_engine, "Status"):
                try:
                    return self._current_engine.Status.RunningState == 2
                except Exception:
                    pass
            return self._is_speaking

    def is_available(self) -> bool:
        """Returns True if the TTS subsystem is operational."""
        return self._available and self._worker_thread.is_alive()

    def stop(self):
        """Gracefully terminate worker thread and stop current speech."""
        self._stop_event.set()
        with self._engine_lock:
            if self._current_engine is not None:
                try:
                    if hasattr(self._current_engine, "Speak"):
                        self._current_engine.Speak("", 3)
                    elif hasattr(self._current_engine, "stop"):
                        self._current_engine.stop()
                except Exception:
                    pass
            self._is_speaking = False


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

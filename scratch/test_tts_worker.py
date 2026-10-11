import time
import queue
import threading
import pyttsx3

class TTSWorker:
    def __init__(self, rate=170, volume=1.0):
        self.rate = rate
        self.volume = volume
        self._queue = queue.PriorityQueue()
        self._is_speaking = False
        self._stop_event = threading.Event()
        self._engine_lock = threading.Lock()
        self._current_engine = None
        self._worker_thread = threading.Thread(target=self._run_loop, daemon=True)
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

            priority, timestamp, text, on_done = item
            print(f"[TTSWorker] Speaking (priority={priority}): {text}")

            self._is_speaking = True
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[TTSWorker] Error during speech: {e}")
            finally:
                self._is_speaking = False
                self._queue.task_done()
                if on_done:
                    try:
                        on_done()
                    except Exception as e:
                        print(f"[TTSWorker] Error in on_done callback: {e}")

    def speak(self, text, priority=10, on_done=None):
        if not text:
            return
        self._queue.put((priority, time.time(), text, on_done))

    def interrupt_and_warn(self, text, on_done=None):
        cleared_items = []
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if item[0] <= 1:
                    cleared_items.append(item)
            except queue.Empty:
                break
        for item in cleared_items:
            self._queue.put(item)

        with self._engine_lock:
            if self._current_engine and self._is_speaking:
                try:
                    self._current_engine.stop()
                except Exception as e:
                    print(f"[TTSWorker] Exception on engine.stop(): {e}")

        self._queue.put((1, time.time(), text, on_done))

    def is_speaking(self):
        return self._is_speaking

    def stop(self):
        self._stop_event.set()
        with self._engine_lock:
            if self._current_engine:
                try:
                    self._current_engine.stop()
                except Exception:
                    pass

if __name__ == "__main__":
    worker = TTSWorker()
    print("Testing 4 rapid interrupt cycles...")
    for i in range(4):
        worker.speak(f"Long normal sentence number {i} talking about the weather and objects.", priority=10)
        time.sleep(0.4)
        worker.interrupt_and_warn(f"Warning {i}! Hazard alert!")
        time.sleep(1.0)
    print("All cycles completed successfully.")

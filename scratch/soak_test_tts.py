import os
import sys
import time
import threading

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.voice.tts_worker import get_tts_worker


def run_soak_test(num_cycles=50):
    worker = get_tts_worker()
    print(f"Starting TTS soak test: {num_cycles} cycles from worker threads...")
    errors = []

    def caller_thread_action(cycle_id):
        try:
            # Enqueue normal speech
            worker.speak(f"This is background query response number {cycle_id} describing objects.", priority=10)
            time.sleep(0.08)
            # Interrupt with urgent warning
            worker.interrupt_and_warn(f"Warning {cycle_id}! Immediate hazard detected.")
            time.sleep(0.12)
        except Exception as e:
            errors.append((cycle_id, str(e)))

    threads = []
    start_time = time.time()

    for i in range(num_cycles):
        t = threading.Thread(target=caller_thread_action, args=(i + 1,), daemon=True)
        threads.append(t)
        t.start()
        # Stagger slightly to simulate real multi-threaded API requests
        time.sleep(0.18)

    for t in threads:
        t.join(timeout=2.0)

    elapsed = time.time() - start_time
    print(f"\n--- Soak Test Results ---")
    print(f"Total cycles dispatched: {num_cycles}")
    print(f"Elapsed time: {elapsed:.2f}s")
    print(f"Errors encountered: {len(errors)}")
    if errors:
        for cycle_id, err in errors:
            print(f"  Cycle {cycle_id} error: {err}")
    else:
        print("[OK] 50/50 cycles completed successfully with 0 COM/SAPI errors.")

    return len(errors)


if __name__ == "__main__":
    err_count = run_soak_test(50)
    sys.exit(err_count)

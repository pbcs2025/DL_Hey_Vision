import threading
import time
import cv2


class LiveCamera:
    """Simple webcam interface for the AI Vision Assistant."""

    def __init__(self, camera_index=0):
        self.camera = cv2.VideoCapture(camera_index)

        if not self.camera.isOpened():
            raise RuntimeError(
                "Could not open the webcam."
            )

    def read_frame(self):
        """Read and return the current webcam frame."""

        success, frame = self.camera.read()

        if not success:
            raise RuntimeError(
                "Could not read a frame from the webcam."
            )

        return frame

    def save_current_frame(
        self,
        output_path="data/test/live_frame.jpg"
    ):
        """Capture and save the current webcam frame."""

        frame = self.read_frame()

        success = cv2.imwrite(
            output_path,
            frame
        )

        if not success:
            raise RuntimeError(
                "Could not save the webcam frame."
            )

        return output_path

    def release(self):
        """Release the webcam."""

        if self.camera is not None:
            self.camera.release()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback
    ):
        self.release()


class WebcamStream:
    """
    Threaded webcam reader for non-blocking real-time Streamlit video streaming.
    Fails gracefully if camera cannot be opened.
    """

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.latest_frame = None
        self.running = False
        self.available = False
        self.lock = threading.Lock()
        self.thread = None
        self.start()

    def start(self):
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.latest_frame = frame
                    self.available = True
                    self.running = True
                    self.thread = threading.Thread(target=self._update_loop, daemon=True)
                    self.thread.start()
                else:
                    self.available = False
            else:
                self.available = False
        except Exception:
            self.available = False

    def _update_loop(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self.available = False
                break
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.latest_frame = frame
                    self.available = True
            else:
                time.sleep(0.01)
            time.sleep(0.01)

    def is_available(self) -> bool:
        return self.available and self.latest_frame is not None

    def get_frame(self):
        """Get latest BGR image frame copy."""
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.available = False
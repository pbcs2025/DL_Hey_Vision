import cv2

from src.vision.live_camera import LiveCamera
from src.voice.speech_to_text import listen
from src.voice.text_to_speech import speak
from src.voice.query_normalizer import normalize_query
from src.pipeline.assistant_pipeline import process_query


FRAME_PATH = "data/test/live_frame.jpg"

EXIT_COMMANDS = [
    "exit",
    "quit",
    "stop",
    "goodbye",
    "close"
]


def run_live_camera_assistant():

    print("\n" + "=" * 60)
    print("AI VISION ASSISTANT - LIVE CAMERA MODE")
    print("=" * 60)

    print("\nStarting webcam...")

    try:
        camera = LiveCamera()
    except RuntimeError as e:
        print(f"❌ {e}")
        return

    print("✅ Webcam started.")

    speak(
        "Live camera mode is ready. "
        "Click the camera window and press V "
        "to ask a question. Press Q to quit."
    )

    try:

        while True:

            success, frame = camera.camera.read()

            if not success:
                print(
                    "❌ Could not read camera frame."
                )
                break

            cv2.imshow(
                "AI Vision Assistant - Live Camera",
                frame
            )

            # Give the camera window keyboard focus.
            key = cv2.waitKey(30) & 0xFF

            # Quit.
            if key in (ord("q"), ord("Q")):

                print(
                    "\nStopping live camera..."
                )

                break

            # Voice question.
            if key in (ord("v"), ord("V")):

                print("\n" + "-" * 60)
                print("🎤 Voice question mode")
                print("-" * 60)

                print(
                    "Listening now..."
                )

                query = listen()

                if not query:

                    print(
                        "⚠️ No usable speech detected."
                    )

                    speak(
                        "Sorry, I could not "
                        "understand you."
                    )

                    continue

                query = query.strip()

                normalized_query = normalize_query(
                    query
                )

                print(
                    f"Original query: {query}"
                )

                print(
                    f"Normalized query: "
                    f"{normalized_query}"
                )

                # Exit through voice.
                if (
                    normalized_query.lower()
                    in EXIT_COMMANDS
                ):

                    speak("Goodbye.")

                    break

                # Capture a fresh frame AFTER
                # the user finishes speaking.
                print(
                    "\n📷 Capturing current "
                    "camera frame..."
                )

                success, frame = camera.camera.read()

                if not success:

                    print(
                        "❌ Could not capture "
                        "current frame."
                    )

                    speak(
                        "Sorry, I could not "
                        "capture the camera view."
                    )

                    continue

                saved = cv2.imwrite(
                    FRAME_PATH,
                    frame
                )

                if not saved:

                    print(
                        "❌ Could not save "
                        "camera frame."
                    )

                    speak(
                        "Sorry, I could not "
                        "process the camera image."
                    )

                    continue

                print(
                    f"✅ Current frame saved: "
                    f"{FRAME_PATH}"
                )

                print(
                    "\n🧠 Processing..."
                )

                result = process_query(
                    normalized_query,
                    FRAME_PATH
                )

                response = result.get(
                    "response",
                    ""
                )

                if not response:

                    response = (
                        "Sorry, I could not "
                        "generate a response."
                    )

                print(
                    f"\n🤖 Assistant: {response}"
                )

                speak(response)

                print(
                    "\n📷 Camera ready again."
                )

                print(
                    "Press V to ask another "
                    "question or Q to quit."
                )

    finally:

        camera.release()

        cv2.destroyAllWindows()

        print(
            "\n✅ Live camera released."
        )

        print(
            "Live camera assistant stopped."
        )


if __name__ == "__main__":
    run_live_camera_assistant()
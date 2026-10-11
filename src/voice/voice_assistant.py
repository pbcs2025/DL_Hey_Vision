from src.voice.speech_to_text import listen
from src.voice.text_to_speech import speak
from src.voice.query_normalizer import normalize_query
from src.pipeline.assistant_pipeline import process_query


IMAGE_PATH = "data/test/test_scene.jpg"


EXIT_COMMANDS = [
    "exit",
    "quit",
    "stop",
    "goodbye",
    "close"
]


def run_voice_assistant():

    print("\n" + "=" * 60)
    print("AI VISION ASSISTANT - VOICE MODE")
    print("=" * 60)

    speak(
        "Hello. I am your AI Vision Assistant. "
        "How can I help you?"
    )

    while True:

        # --------------------------------------------------
        # 1. Listen to the user
        # --------------------------------------------------
        query = listen()

        # --------------------------------------------------
        # 2. Handle empty / unrecognized speech
        # --------------------------------------------------
        if not query:

            print(
                "⚠️ No usable speech detected."
            )

            speak(
                "Sorry, I could not understand you. "
                "Please try again."
            )

            continue

        # --------------------------------------------------
        # 3. Clean the query
        # --------------------------------------------------
        query = query.strip()

        # --------------------------------------------------
        # 4. Correct common speech-to-text mistakes
        # --------------------------------------------------
        normalized_query = normalize_query(query)

        print(
            f"Original query: {query}"
        )

        print(
            f"Normalized query: {normalized_query}"
        )

        # --------------------------------------------------
        # 5. Check exit commands
        # --------------------------------------------------
        if normalized_query.lower() in EXIT_COMMANDS:

            speak("Goodbye.")

            print(
                "\nVoice assistant stopped."
            )

            break

        # --------------------------------------------------
        # 6. Process the query
        # --------------------------------------------------
        print("\n" + "-" * 60)
        print(
            f"User query: {normalized_query}"
        )
        print("-" * 60)

        print("🧠 Processing...")

        result = process_query(
            normalized_query,
            IMAGE_PATH
        )

        # --------------------------------------------------
        # 7. Get AI response
        # --------------------------------------------------
        response = result.get(
            "response",
            ""
        )

        if not response:

            response = (
                "Sorry, I could not generate "
                "a response."
            )

        print(
            f"\n🤖 Assistant: {response}"
        )

        # --------------------------------------------------
        # 8. Speak the response
        # --------------------------------------------------
        speak(response)


if __name__ == "__main__":
    run_voice_assistant()
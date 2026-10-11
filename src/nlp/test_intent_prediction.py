import numpy as np
import tensorflow as tf


MODEL_PATH = "models/nlp/query_embedding_model_expanded.keras"
INTENT_PATH = "models/nlp/intent_names.npy"


def predict_intent(model, intent_names, query):
    prediction = model.predict(
        tf.constant([query], dtype=tf.string),
        verbose=0
    )

    predicted_id = int(np.argmax(prediction[0]))
    confidence = float(prediction[0][predicted_id])

    return intent_names[predicted_id], confidence


def main():
    print("\nAI Vision Assistant - NLP Intent Prediction Test")
    print("=" * 60)

    print("\nLoading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    intent_names = np.load(
        INTENT_PATH,
        allow_pickle=True
    )

    test_queries = [
        "What do I have around me?",
        "What is directly ahead?",
        "Where is the door?",
        "Can you read this sign?",
        "Is someone coming toward me?",
        "Is there something blocking my path?",
        "What am I looking at?",
        "Describe this scene",
        "Say that again",
        "Tell me more"
    ]

    print("\nPredictions")
    print("=" * 60)

    for query in test_queries:
        intent, confidence = predict_intent(
            model,
            intent_names,
            query
        )

        print(f"\nQuery: {query}")
        print(f"Predicted Intent: {intent}")
        print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
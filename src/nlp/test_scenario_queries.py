import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from .intent_labels import INTENTS


MODEL_PATH = "models/nlp/query_embedding_model_expanded.keras"

model = load_model(MODEL_PATH, compile=False)


test_queries = {
    "SCENE_SUMMARY": [
        "What is around me?",
        "Tell me what is around me",
        "What can you see around me?",
        "What is in my surroundings?"
    ],

    "READ_TEXT": [
        "Read this sign",
        "What does this sign say?",
        "Can you read the text?"
    ],

    "DETECT_APPROACHING": [
        "Is anything approaching me?",
        "Is someone coming toward me?",
        "Is anything moving closer to me?"
    ],

    "OBJECT_IDENTIFICATION": [
        "What is this?",
        "What object is this?",
        "Can you identify this object?"
    ],

    "SCENE_DESCRIPTION": [
        "Give me a quick description of the scene",
        "Describe the scene briefly",
        "What does the scene look like?"
    ],

    "MORE_DETAIL": [
        "Tell me more",
        "Can you provide more details?",
        "Explain that in more detail"
    ]
}


print("=" * 70)
print("NLP SCENARIO QUERY EVALUATION")
print("=" * 70)

correct = 0
total = 0

for expected_intent, queries in test_queries.items():

    print(f"\n--- Expected: {expected_intent} ---")

    predictions = model.predict(
        tf.constant(queries),
        verbose=0
    )

    for query, prediction in zip(queries, predictions):

        intent_id = int(np.argmax(prediction))
        predicted_intent = INTENTS[intent_id]
        confidence = float(prediction[intent_id])

        status = "PASS" if predicted_intent == expected_intent else "FAIL"

        if status == "PASS":
            correct += 1

        total += 1

        print(
            f"{status:4s} | "
            f"{confidence:.4f} | "
            f"{predicted_intent:22s} | "
            f"{query}"
        )


accuracy = correct / total * 100

print("\n" + "=" * 70)
print(f"Scenario Query Accuracy: {accuracy:.2f}%")
print(f"Passed: {correct}/{total}")
print("=" * 70)
import numpy as np
import tensorflow as tf

from src.dqn.dqn_bridge import get_dqn_action
from .intent_labels import INTENTS


# --------------------------------------------------
# Model
# --------------------------------------------------

MODEL_PATH = "models/nlp/query_embedding_model_expanded.keras"

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)




# --------------------------------------------------
# Query priority
# --------------------------------------------------

QUERY_PRIORITIES = {
    "SCENE_SUMMARY": 0.4,
    "FRONT_OBJECTS": 0.7,
    "LOCATE_OBJECT": 0.8,
    "READ_TEXT": 0.8,
    "DETECT_APPROACHING": 1.0,
    "OBSTACLE_CHECK": 1.0,
    "OBJECT_IDENTIFICATION": 0.5,
    "SCENE_DESCRIPTION": 0.4,
    "REPEAT_RESPONSE": 0.3,
    "MORE_DETAIL": 0.5
}


# --------------------------------------------------
# Predict intent
# --------------------------------------------------

def predict_intent(query):

    prediction = model.predict(
        tf.constant([query]),
        verbose=0
    )[0]

    intent_id = int(
        np.argmax(prediction)
    )

    confidence = float(
        prediction[intent_id]
    )

    return (
        INTENTS[intent_id],
        confidence
    )


# --------------------------------------------------
# Test queries
# --------------------------------------------------

test_queries = [
    "Is anything approaching me?",
    "Is there an obstacle ahead?",
    "Where is the door?",
    "What is in front of me?",
    "What is this?",
    "Give me a quick description of the scene"
]


print()
print("AI Vision Assistant")
print("NLP → Intent → DQN")
print("=" * 65)


for query in test_queries:

    intent, intent_confidence = predict_intent(
        query
    )

    priority = QUERY_PRIORITIES.get(
        intent,
        0.5
    )

    # Simulated visual state.
    # The purpose here is only to verify
    # that NLP priority reaches the DQN.
    decision = get_dqn_action(
        risk=0.33,
        confidence=0.85,
        movement="stationary",
        ahead=1,
        intent=intent,
        recent_alert=0
    )

    print()
    print(f"Query: {query}")
    print(
        f"Intent: {intent} "
        f"({intent_confidence:.4f})"
    )
    print(f"Priority: {priority:.2f}")
    print(f"DQN decision: {decision['action']}")
    print(f"DQN Q-values: {decision['q_values']}")


print()
print("NLP + DQN integration test completed.")
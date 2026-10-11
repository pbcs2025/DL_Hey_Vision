from src.pipeline.assistant_pipeline import process_query


test_cases = [
    "What is around me?",
    "What is in front of me?",
    "Where is the door?",
    "Read this sign.",
    "Is anything approaching me?",
    "Is there an obstacle ahead?",
    "What is this?",
    "Give me a quick description of the scene."
]


print("=" * 75)
print("AI VISION ASSISTANT - END-TO-END SCENARIO TEST")
print("=" * 75)

for i, query in enumerate(test_cases, start=1):

    print("\n" + "-" * 75)
    print(f"SCENARIO {i}")
    print(f"Query: {query}")
    print("-" * 75)

    try:
        result = process_query(
            query,
            "data/test/door_test.png"
        )

        print(f"Intent: {result['intent']}")
        print(f"Intent confidence: {result['intent_confidence']:.4f}")

        print(f"Response: {result['response']}")

        print("DQN Action: Handled internally by the pipeline")

    except Exception as e:
        print(f"ERROR: {e}")

print("\n" + "=" * 75)
print("END-TO-END TEST COMPLETED")
print("=" * 75)
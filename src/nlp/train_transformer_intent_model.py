import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .transformer_intent_model import (
    build_vectorizer,
    build_transformer_intent_model
)


DATA_PATH = "data/custom/queries/query_dataset_expanded.csv"
MODEL_PATH = "models/nlp/transformer_intent_model_expanded.keras"
INTENT_PATH = "models/nlp/intent_names_transformer.npy"


# Load dataset
df = pd.read_csv(DATA_PATH)

texts = df["text"].astype(str).values
intent_names = sorted(df["intent"].unique())

intent_to_id = {
    intent: i for i, intent in enumerate(intent_names)
}

labels = np.array([
    intent_to_id[intent]
    for intent in df["intent"]
])


print("\nTransformer Intent Dataset")
print("=" * 50)
print("Total samples:", len(texts))
print("Number of intents:", len(intent_names))


print("\nIntent Mapping")
print("=" * 50)

for i, intent in enumerate(intent_names):
    print(f"{i} -> {intent}")


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)


print("\nDataset Split")
print("=" * 50)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# Build and adapt vectorizer
vectorizer = build_vectorizer()

vectorizer.adapt(
    tf.constant(X_train, dtype=tf.string)
)

print("\nVocabulary")
print("=" * 50)
print("Vocabulary size:", len(vectorizer.get_vocabulary()))


# Build Transformer
model = build_transformer_intent_model(vectorizer)

# Build model
model(tf.constant(["test query"]))

print("\nTransformer Intent Model")
print("=" * 50)
print("Parameters:", model.count_params())


# Train
print("\nTraining Transformer Intent Model")
print("=" * 50)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=30,
    batch_size=16,
    verbose=1
)


# Evaluate
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)


print("\nFinal Evaluation")
print("=" * 50)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# Save model
model.save(MODEL_PATH)

print("\nModel saved to:")
print(MODEL_PATH)


# Save intent mapping
np.save(
    INTENT_PATH,
    np.array(intent_names)
)

print("\nIntent mapping saved to:")
print(INTENT_PATH)
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .embedding_model import build_vectorizer, build_embedding_model


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

DATA_PATH = "data/custom/queries/query_dataset_expanded.csv"    

df = pd.read_csv(DATA_PATH)

print("\nQuery Dataset")
print("=" * 40)
print("Total samples:", len(df))
print("Number of intents:", df["intent"].nunique())


# --------------------------------------------------
# 2. Convert intents to integer labels
# --------------------------------------------------

intent_names = sorted(df["intent"].unique())

intent_to_id = {
    intent: index
    for index, intent in enumerate(intent_names)
}

id_to_intent = {
    index: intent
    for intent, index in intent_to_id.items()
}

df["label"] = df["intent"].map(intent_to_id)

print("\nIntent Mapping")
print("=" * 40)

for index, intent in id_to_intent.items():
    print(index, "->", intent)


# --------------------------------------------------
# 3. Prepare text and labels
# --------------------------------------------------

texts = df["text"].values
labels = df["label"].values


# --------------------------------------------------
# 4. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

print("\nDataset Split")
print("=" * 40)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 5. Build and adapt vectorizer
# --------------------------------------------------

vectorizer = build_vectorizer()

vectorizer.adapt(X_train)

print("\nVocabulary")
print("=" * 40)
print("Vocabulary size:", len(vectorizer.get_vocabulary()))


# --------------------------------------------------
# 6. Build embedding model
# --------------------------------------------------

num_classes = len(intent_names)

model = build_embedding_model(
    vectorizer,
    num_classes
)

# Build the model by passing one sample
model(tf.constant(["Where is the door?"]))

print("\nEmbedding Model")
print("=" * 40)
print("Parameters:", model.count_params())


# --------------------------------------------------
# 7. Train model
# --------------------------------------------------

print("\nTraining Embedding Model")
print("=" * 40)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=30,
    batch_size=16,
    verbose=1
)


# --------------------------------------------------
# 8. Evaluate model
# --------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nFinal Evaluation")
print("=" * 40)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


# --------------------------------------------------
# 9. Save model and intent mapping
# --------------------------------------------------

MODEL_PATH = "models/nlp/query_embedding_model_expanded.keras"

model.save(MODEL_PATH)

np.save(
    "models/nlp/intent_names.npy",
    np.array(intent_names)
)

print("\nModel saved to:")
print(MODEL_PATH)

print("\nIntent mapping saved to:")
print("models/nlp/intent_names.npy")
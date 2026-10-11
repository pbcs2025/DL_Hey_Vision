import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from .lstm_intent_model import build_vectorizer


DATA_PATH = "data/custom/queries/query_dataset_expanded.csv"
MODEL_PATH = "models/nlp/lstm_intent_model_expanded.keras"
INTENT_PATH = "models/nlp/intent_names_lstm.npy"


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


# Same split used during training
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)


print("\nLSTM Intent Model Evaluation")
print("=" * 50)
print("Total samples:", len(texts))
print("Testing samples:", len(X_test))


# Load trained model
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"TextVectorization": tf.keras.layers.TextVectorization}
)


# Predict
predictions = model.predict(
    tf.constant(X_test, dtype=tf.string),
    verbose=0
)

y_pred = np.argmax(predictions, axis=1)


# Accuracy
accuracy = np.mean(y_pred == y_test)

print("\nOverall Accuracy")
print("=" * 50)
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")


# Classification report
print("\nClassification Report")
print("=" * 50)

print(
    classification_report(
        y_test,
        y_pred,
        labels=list(range(len(intent_names))),
        target_names=intent_names,
        digits=4
    )
)


# Confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=list(range(len(intent_names)))
)

print("\nConfusion Matrix")
print("=" * 50)

cm_df = pd.DataFrame(
    cm,
    index=intent_names,
    columns=intent_names
)

print(cm_df)
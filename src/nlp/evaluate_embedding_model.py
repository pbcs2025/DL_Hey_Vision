import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

DATA_PATH = "data/custom/queries/query_dataset_expanded.csv"
MODEL_PATH = "models/nlp/query_embedding_model_expanded.keras"
INTENT_PATH = "models/nlp/intent_names.npy"


def main():
    print("\nAI Vision Assistant - NLP Embedding Model Evaluation")
    print("=" * 65)

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------
    df = pd.read_csv(DATA_PATH)

    texts = df["text"].astype(str).values
    intents = df["intent"].values

    intent_names = np.load(
        INTENT_PATH,
        allow_pickle=True
    )

    intent_to_id = {
        intent: index
        for index, intent in enumerate(intent_names)
    }

    labels = np.array([
        intent_to_id[intent]
        for intent in intents
    ])

    print("\nDataset")
    print("=" * 65)
    print(f"Total samples: {len(texts)}")
    print(f"Number of intents: {len(intent_names)}")

    # ---------------------------------------------------------
    # Recreate the same stratified test split
    # ---------------------------------------------------------
    _, X_test, _, y_test = train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels
    )

    print("\nTest Set")
    print("=" * 65)
    print(f"Test samples: {len(X_test)}")

    # ---------------------------------------------------------
    # Load trained model
    # ---------------------------------------------------------
    print("\nLoading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # ---------------------------------------------------------
    # Predict
    # ---------------------------------------------------------
    predictions = model.predict(
        tf.constant(X_test, dtype=tf.string),
        verbose=0
    )

    y_pred = np.argmax(
        predictions,
        axis=1
    )

    # ---------------------------------------------------------
    # Overall accuracy
    # ---------------------------------------------------------
    accuracy = np.mean(
        y_pred == y_test
    )

    print("\nOverall Performance")
    print("=" * 65)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # ---------------------------------------------------------
    # Classification report
    # ---------------------------------------------------------
    print("\nClassification Report")
    print("=" * 65)

    report = classification_report(
        y_test,
        y_pred,
        target_names=intent_names,
        digits=4
    )

    print(report)

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------
    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\nConfusion Matrix")
    print("=" * 65)

    print(
        pd.DataFrame(
            cm,
            index=intent_names,
            columns=intent_names
        )
    )


if __name__ == "__main__":
    main()
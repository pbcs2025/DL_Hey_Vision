from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input


def build_risk_model():
    """
    Build an ANN for prototype risk-priority classification.

    Input features:
        object type
        position
        approximate visual proximity
        detection confidence
        movement
        ahead indicator

    Output:
        LOW
        MEDIUM
        HIGH
        CRITICAL
    """

    model = Sequential([
        Input(shape=(6,)),

        Dense(32, activation="relu"),

        Dropout(0.2),

        Dense(16, activation="relu"),

        Dense(4, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
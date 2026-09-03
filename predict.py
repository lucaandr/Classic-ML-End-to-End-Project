import joblib
import numpy as np


def load_model(model_path="random_forest_fraud_model.pkl"):

    return joblib.load(model_path)


def predict_transaction(
    features: list, model=None, threshold: float = 0.15
) -> dict:
    if model is None:
        model = load_model()

    data_array = np.array(features).reshape(1, -1)
    fraud_probability = model.predict_proba(data_array)[0, 1]
    is_fraud = bool(fraud_probability >= threshold)

    return {
        "is_fraud": is_fraud,
        "fraud_probability": float(fraud_probability),
        "decision": "BLOCATA (Frauda)" if is_fraud else "APROBATĂ (Legitim)",
    }


if __name__ == "__main__":
    rf_model = load_model()
    dummy_input = [0.0] * rf_model.n_features_in_
    result = predict_transaction(dummy_input, model=rf_model)

    print("Rezultat test inferenta:", result)

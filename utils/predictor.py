"""
=====================================================
Customer Churn Prediction
utils/predictor.py

Loads the trained pipeline (model/model.pkl, produced by
EDA.ipynb) and exposes a simple .predict(customer_dict)
API used by app.py.

The model was trained on a REDUCED set of 10 features
(selected by feature importance in EDA.ipynb) instead of
all 19 original columns, to keep the front-end form short.
=====================================================
"""

import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join("model", "model.pkl")

# Must match the FEATURES list used to train the model in EDA.ipynb.
FEATURE_COLUMNS = [
    "tenure",
    "Contract",
    "MonthlyCharges",
    "TotalCharges",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "PaymentMethod",
    "PaperlessBilling",
    "MultipleLines",
]


class ChurnPredictor:
    """Thin wrapper around the trained scikit-learn pipeline."""

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. "
                "Run EDA.ipynb end-to-end in VS Code first — it trains the "
                "model and saves model/model.pkl."
            )
        self.model = joblib.load(model_path)

    def predict(self, customer: dict) -> dict:
        """
        customer: dict of raw form values (see FEATURE_COLUMNS).
        Returns: {"prediction": 0/1, "probability": float, "risk": str}
        """
        row = pd.DataFrame([customer], columns=FEATURE_COLUMNS)

        pred = int(self.model.predict(row)[0])
        proba = float(self.model.predict_proba(row)[0][1])  # probability of churn

        if proba >= 0.7:
            risk = "High Risk"
        elif proba >= 0.4:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

        return {
            "prediction": pred,
            "probability": round(proba * 100, 2),
            "risk": risk,
        }

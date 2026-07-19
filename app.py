"""
=====================================================
Customer Churn Prediction
Flask Application
=====================================================
"""

from flask import Flask, render_template, request
import pandas as pd
import os

from utils.predictor import ChurnPredictor

app = Flask(__name__)

# Load Predictor
predictor = ChurnPredictor()

# Prediction History File
PREDICTION_FILE = "saved_predictions/predictions.csv"

os.makedirs("saved_predictions", exist_ok=True)


# ===============================
# Home Page
# ===============================
@app.route("/")
def home():
    return render_template("index.html")


# ===============================
# Prediction
# ===============================
@app.route("/predict", methods=["POST"])
def predict():

    # Only the 10 reduced features (selected by feature importance
    # in EDA.ipynb) are collected from the form.
    customer = {

        "tenure": int(request.form["tenure"]),
        "Contract": request.form["Contract"],
        "MonthlyCharges": float(request.form["MonthlyCharges"]),
        "TotalCharges": float(request.form["TotalCharges"]),
        "InternetService": request.form["InternetService"],
        "OnlineSecurity": request.form["OnlineSecurity"],
        "TechSupport": request.form["TechSupport"],
        "PaymentMethod": request.form["PaymentMethod"],
        "PaperlessBilling": request.form["PaperlessBilling"],
        "MultipleLines": request.form["MultipleLines"],

    }

    result = predictor.predict(customer)

    history = customer.copy()

    history["Prediction"] = (
        "Churn"
        if result["prediction"] == 1
        else "No Churn"
    )

    history["Probability"] = result["probability"]

    history["Risk"] = result["risk"]

    df = pd.DataFrame([history])

    if os.path.exists(PREDICTION_FILE):

        df.to_csv(
            PREDICTION_FILE,
            mode="a",
            header=False,
            index=False
        )

    else:

        df.to_csv(
            PREDICTION_FILE,
            index=False
        )

    return render_template(

        "result.html",

        prediction=history["Prediction"],

        probability=result["probability"],

        risk=result["risk"]

    )


# ===============================
# Dashboard
# ===============================
@app.route("/dashboard")
def dashboard():

    if os.path.exists(PREDICTION_FILE):

        df = pd.read_csv(PREDICTION_FILE)

        total_predictions = len(df)

        churn_count = len(df[df["Prediction"] == "Churn"])

        no_churn_count = len(df[df["Prediction"] == "No Churn"])

        avg_probability = round(
            df["Probability"].mean(),
            2
        )

    else:

        total_predictions = 0
        churn_count = 0
        no_churn_count = 0
        avg_probability = 0

    return render_template(

        "dashboard.html",

        total_predictions=total_predictions,

        churn_count=churn_count,

        no_churn_count=no_churn_count,

        avg_probability=avg_probability

    )


# ===============================
# Analytics Page
# ===============================
@app.route("/analytics")
def analytics():

    if os.path.exists(PREDICTION_FILE):

        df = pd.read_csv(PREDICTION_FILE)

        tables = df.tail(20).to_html(
            classes="table table-striped",
            index=False
        )

    else:

        tables = "<h3>No Prediction History Available</h3>"

    return render_template(

        "analytics.html",

        tables=tables

    )


# ===============================
# Main
# ===============================
if __name__ == "__main__":

    app.run(

        debug=True,

        host="0.0.0.0",

        port=5000

    )

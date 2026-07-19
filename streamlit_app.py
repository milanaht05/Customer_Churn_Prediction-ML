"""
=====================================================
Customer Churn Prediction — Streamlit App
=====================================================
Run with:  streamlit run streamlit_app.py

Loads model/model.pkl, which is trained and saved by
EDA.ipynb using a REDUCED set of 10 features (chosen by
feature importance) instead of all 19 original columns.
"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered",
)

# ---------------------------------------------------
# Mild / soft color theme (non-white background)
# ---------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(160deg, #eaf1fb 0%, #eef7f2 100%);
    }
    h1, h2, h3 { color: #3c4656; }
    section[data-testid="stForm"] {
        background-color: #f3f7fc;
        border: 1px solid #d9e3f0;
        border-radius: 14px;
        padding: 1.4em;
    }
    div.stButton > button {
        background-color: #7fa8d9;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.6em 1.4em;
        font-weight: 700;
    }
    div.stButton > button:hover { background-color: #5f8ec9; color: white; }
    .result-box {
        padding: 1.2em;
        border-radius: 14px;
        text-align: center;
        margin-top: 1em;
    }
    .churn-box { background-color: #fbebeb; border: 1px solid #e6a1a1; color: #a15b5b; }
    .no-churn-box { background-color: #eaf6f0; border: 1px solid #a8d5c2; color: #4d8a71; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------
# Load model
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model/model.pkl")


model = load_model()

st.title("📊 Customer Churn Prediction")
st.caption("Just 10 key details are needed to estimate churn risk.")

# ---------------------------------------------------
# Input form (reduced feature set)
# ---------------------------------------------------
with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input("Tenure (months)", 0, 100, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 500.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 840.0)
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    with col2:
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

    submitted = st.form_submit_button("Predict Churn")

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
if submitted:
    row = pd.DataFrame([{
        "tenure": tenure,
        "Contract": contract,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "TechSupport": tech_support,
        "PaymentMethod": payment_method,
        "PaperlessBilling": paperless_billing,
        "MultipleLines": multiple_lines,
    }])

    pred = int(model.predict(row)[0])
    proba = float(model.predict_proba(row)[0][1]) * 100

    if proba >= 70:
        risk = "High Risk"
    elif proba >= 40:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    box_class = "churn-box" if pred == 1 else "no-churn-box"
    label = "Churn" if pred == 1 else "No Churn"

    st.markdown(
        f"""
        <div class="result-box {box_class}">
            <h2>{label}</h2>
            <p><b>{proba:.2f}%</b> probability of churn — {risk}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(min(int(proba), 100))

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ------------------------
# Load model, scaler, features
# ------------------------
model = pickle.load(open("model/churn_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
features = pickle.load(open("model/features.pkl", "rb"))

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.write("Predict whether a customer is likely to churn")

# ------------------------
# User inputs
# ------------------------
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# ------------------------
# Create full feature vector
# ------------------------
input_data = pd.DataFrame(np.zeros((1, len(features))), columns=features)

# Fill user inputs
input_data['tenure'] = tenure
input_data['MonthlyCharges'] = monthly_charges
input_data['TotalCharges'] = total_charges

# Contracts
if 'Contract_One year' in input_data.columns:
    input_data['Contract_One year'] = 1 if contract == "One year" else 0
if 'Contract_Two year' in input_data.columns:
    input_data['Contract_Two year'] = 1 if contract == "Two year" else 0

# Internet service
if 'InternetService_Fiber optic' in input_data.columns:
    input_data['InternetService_Fiber optic'] = 1 if internet_service == "Fiber optic" else 0
if 'InternetService_No' in input_data.columns:
    input_data['InternetService_No'] = 1 if internet_service == "No" else 0

# ------------------------
# Scale input & predict
# ------------------------
input_scaled = scaler.transform(input_data)

if st.button("🔍 Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to churn (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Customer is not likely to churn (Probability: {probability:.2%})")

    # Optional: show probability bar
    st.progress(int(probability * 100))

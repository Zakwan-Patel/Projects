import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("fraud_model.pkl", "rb"))

st.title("💳 Fraud Detection System")

features = []

for i in range(1, 29):
    val = st.number_input(f"Feature V{i}", value=0.0)
    features.append(val)

amount = st.number_input("Transaction Amount", value=0.0)
features.append(amount)

if st.button("Predict"):
    prediction = model.predict([features])[0]
    
    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction")
    else:
        st.success("✅ Legitimate Transaction")
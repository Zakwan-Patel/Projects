import streamlit as st
import pandas as pd
import pickle

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf.pkl", "rb"))

st.title("📰 Fake News Detection")

input_text = st.text_area("Enter News Article:")

if st.button("Predict"):
    if input_text:
        transformed = vectorizer.transform([input_text])
        prediction = model.predict(transformed)[0]
        
        if prediction == 1:
            st.success("✅ Real News")
        else:
            st.error("❌ Fake News")
import streamlit as st
from transformers import pipeline

summarizer = pipeline("summarization")

st.title("✍️ AI Text Summarizer")

text = st.text_area("Enter Long Text:")

if st.button("Summarize"):
    if text:
        summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
        st.write(summary[0]['summary_text'])
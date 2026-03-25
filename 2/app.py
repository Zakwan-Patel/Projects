import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import pickle

vectorizer = pickle.load(open("tfidf.pkl", "rb"))

st.title("📄 Resume Ranker")

job_desc = st.text_area("Enter Job Description:")
resumes = st.text_area("Paste Multiple Resumes (separate by ---):")

if st.button("Rank"):
    resume_list = resumes.split("---")
    
    all_docs = [job_desc] + resume_list
    vectors = vectorizer.transform(all_docs)
    
    scores = cosine_similarity(vectors[0:1], vectors[1:])[0]
    
    ranked = sorted(zip(resume_list, scores), key=lambda x: x[1], reverse=True)
    
    for i, (res, score) in enumerate(ranked):
        st.write(f"Rank {i+1} - Score: {score:.2f}")
        st.write(res[:200] + "...")
        st.write("---")
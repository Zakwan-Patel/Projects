import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv("2\Resume.csv")

# Use resume text
texts = df['Resume_str'].dropna().tolist()

# Train vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
vectorizer.fit(texts)

# Save
pickle.dump(vectorizer, open("tfidf.pkl", "wb"))
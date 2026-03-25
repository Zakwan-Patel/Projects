import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load data
df = pd.read_csv("tmdb_top_rated_by_genre_platforms.csv")
df['overview'] = df['overview'].fillna('')

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])

# Cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Index mapping
indices = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()

# Recommendation function
def get_recommendations(title, media_type, top_n=5):
    title = title.lower()
    
    # Find closest match
    matches = df[df['title'].str.lower().str.contains(title)]
    
    if matches.empty:
        return None
    
    idx = matches.index[0]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:30]
    
    movie_indices = [i[0] for i in sim_scores]
    
    recommendations = df.iloc[movie_indices]
    recommendations = recommendations[recommendations['media_type'] == media_type]
    
    return recommendations[['title', 'overview', 'poster_path', 'platforms']].head(top_n)

# UI
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

st.title("🎬 Movie Recommendation System")

movie_name = st.selectbox("Select Movie:", df['title'].values)
media_type = st.selectbox("Select Type", ["movie", "tv"])

if st.button("Recommend"):
    results = get_recommendations(movie_name, media_type)
    
    if results is None or results.empty:
        st.error("Movie not found. Try another one.")
    else:
        for _, row in results.iterrows():
            st.subheader(row['title'])
            
            if pd.notna(row['poster_path']):
                st.image(f"https://image.tmdb.org/t/p/w500{row['poster_path']}")
            
            st.write(row['overview'])
            st.write(f"📺 Available on: {row['platforms']}")
            st.write("---")
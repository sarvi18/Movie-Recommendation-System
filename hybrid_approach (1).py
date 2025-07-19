import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
tags = pd.read_csv("tags.csv")

def extract_year(title):
    match = re.search(r"\((\d{4})\)", title)
    return int(match.group(1)) if match else None

movies['year'] = movies['title'].apply(extract_year)
movies['genres'] = movies['genres'].replace("(no genres listed)", "").fillna("")
movies['year'] = movies['year'].fillna(movies['year'].median())

genre_tfidf = TfidfVectorizer(tokenizer=lambda x: x.split('|'))
genre_matrix = genre_tfidf.fit_transform(movies['genres'])

title_tfidf = TfidfVectorizer()
title_matrix = title_tfidf.fit_transform(movies['title'])

scaler = MinMaxScaler()
year_scaled = scaler.fit_transform(movies[['year']])

total_content_features = np.hstack([genre_matrix.toarray(), title_matrix.toarray(), year_scaled])
content_similarity = cosine_similarity(total_content_features)

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

svd_model = SVD()
svd_model.fit(trainset)

def hybrid_recommendations(user_id, top_n=10, alpha=0.6):
    user_rated_movies = ratings[ratings['userId'] == user_id]['movieId'].tolist()

    scores = []
    for idx, row in movies.iterrows():
        movie_id = row['movieId']
        if movie_id in user_rated_movies:
            continue
        try:
            collab_score = svd_model.predict(user_id, movie_id).est
        except:
            collab_score = 3.0

        rated_indices = movies[movies['movieId'].isin(user_rated_movies)].index
        sim_scores = content_similarity[idx][rated_indices]
        content_score = sim_scores.mean() if len(sim_scores) > 0 else 0

        content_score_scaled = 0.5 + content_score * 4.5
        hybrid = alpha * collab_score + (1 - alpha) * content_score_scaled
        scores.append((movie_id, row['title'], hybrid))

    ranked = sorted(scores, key=lambda x: x[2], reverse=True)
    return ranked[:top_n]

recommendations = hybrid_recommendations(user_id=1, top_n=10)
for movie_id, title, score in recommendations:
    print(f"{title} (Score: {score:.2f})")

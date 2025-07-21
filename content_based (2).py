import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, hstack

movies = pd.read_csv("movies.csv")
tags = pd.read_csv("tags.csv")

def extract_year(title):
    match = re.search(r"\((\d{4})\)", title)
    return int(match.group(1)) if match else None

movies['year'] = movies['title'].apply(extract_year)
movies['year'] = movies['year'].fillna(movies['year'].median())

genre_tfidf = TfidfVectorizer(tokenizer=lambda x: x.split('|'))
genre_matrix = genre_tfidf.fit_transform(movies['genres'])

title_tfidf = TfidfVectorizer()
title_matrix = title_tfidf.fit_transform(movies['title'])

scaler = MinMaxScaler()
year_matrix = scaler.fit_transform(movies[['year']])
year_matrix = csr_matrix(year_matrix)

tag_matrix = tags.groupby(['movieId', 'tag']).size().unstack(fill_value=0)
tag_matrix = tag_matrix.reindex(movies['movieId']).fillna(0)
tag_matrix = tag_matrix.div(tag_matrix.max(axis=1), axis=0).fillna(0)
tag_matrix = csr_matrix(tag_matrix.values)

genre_weight = 0.4
year_weight = 0.2
tag_weight = 0.1
title_weight=0.3

genre_matrix = genre_matrix * genre_weight
year_matrix = year_matrix * year_weight
tag_matrix = tag_matrix * tag_weight
title_matrix = title_matrix * title_weight

combined_features = hstack([genre_matrix, year_matrix, tag_matrix, title_matrix])
similarity = cosine_similarity(combined_features)

def recommend(movie_title, top_n=10):
    idx = movies[movies['title'].str.contains(movie_title, case=False, na=False)].index
    if len(idx) == 0:
        return "Movie not found"
    idx = idx[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:top_n+1]]
    return movies.iloc[top_indices][['movieId', 'title', 'genres', 'year']]

recommend("Toy Story", top_n=10)
recommend("Batman", top_n=10)

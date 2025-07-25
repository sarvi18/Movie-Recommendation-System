# Movie-Recommendation-System
# Movie/Show Recommendation and Analysis System

This repository contains code and documentation for a machine learning-based system that performs intelligent movie/show recommendations, genre classification, and popularity prediction using metadata.

The project explores three primary tasks:

1. Movie Recommendation
   - Collaborative Filtering using user ratings
   - Content-Based Filtering using metadata (genre, cast, director, etc.)
   - Hybrid Approach combining the above two methods

2. Genre Classification
   - Predicts one or more genres for a given movie using metadata
   - Multi-label classification using Random Forest and vectorized features

3. Popularity Prediction
   - Predicts a movie's score (e.g., IMDb rating) based on metadata using a regression model

# Repository Structure

- collaborative.py # Collaborative filtering using Surprise
     - ratings.csv
     - movies.csv # Datasets used

- content_based (2).py # Content-based filtering using Cosine Similarity
     - movies.csv
     - tags.csv

- hybrid_approach (1).py # Hybrid model combining both
     - ratings.csv
     - movies.csv
     - tags.csv

- genre_classification (2).py # Genre classifier using multi label classification
     - netflix_titles.csv

- popularity_prediction.py # Popularity prediction using regression model
     - movie.csv

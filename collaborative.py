from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import pandas as pd

ratings_df = pd.read_csv("ratings.csv")
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

print(ratings_df.info())
print(ratings_df.head())
print("\nNull values:\n", ratings_df.isnull().sum())
print("\nDuplicate rows:", ratings_df.duplicated().sum())
print("\nRating range:", ratings_df['rating'].min(), "to", ratings_df['rating'].max())

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

model = SVD()
model.fit(trainset)

predictions = model.test(testset)
accuracy.rmse(predictions)

movies_df=pd.read_csv("movies.csv")

print(movies_df.info())
print(movies_df.head())
print("\nNull values:\n", movies_df.isnull().sum())
print("\nDuplicate rows:", movies_df.duplicated().sum())
print("\nUnique movieIds:", movies_df['movieId'].nunique(), " / Total rows:", len(movies_df))
print("\nMissing titles:", movies_df['title'].isnull().sum())
print("Missing genres:", movies_df['genres'].isnull().sum())

user_id = 1
rated_movies = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()
all_movies = movies_df['movieId'].unique()

unrated_movies = [m for m in all_movies if m not in rated_movies]
from surprise import PredictionImpossible
predictions = []
for movie_id in unrated_movies:
    try:
        pred = model.predict(user_id, movie_id)
        predictions.append((movie_id, pred.est))
    except PredictionImpossible:
        continue

top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:10]
for movie_id, predicted_rating in top_n:
    title = movies_df[movies_df['movieId'] == movie_id]['title'].values[0]
    print(f"{title} — predicted rating: {predicted_rating:.2f}")

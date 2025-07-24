import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import category_encoders as ce

df = pd.read_csv("movie.csv")

df = df[['score', 'star', 'name', 'year', 'genre', 'director']]

df['star'] = df['star'].fillna('Unknown_Star')
df['name'] = df['name'].fillna('Unknown_Name')
df['year'] = df['year'].fillna(df['year'].median())
df['genre'] = df['genre'].fillna('Unknown_Genre')
df['director'] = df['director'].fillna('Unknown_Director')
df['score']=df['score'].fillna(df['score'].median())

X = df.drop(columns=['score'])
y = df['score']

preprocessor = ColumnTransformer(transformers=[
    ('genre', TfidfVectorizer(tokenizer=lambda x: x.split('|')), 'genre'),
], remainder='passthrough')

target_enc = ce.TargetEncoder(cols=['star', 'director', 'name'])
X_encoded = target_enc.fit_transform(X[['star', 'director', 'name']], y)

X_temp = X.copy()
X_temp[['star', 'director', 'name']] = X_encoded

full_preprocessor = ColumnTransformer(transformers=[
    ('genre', TfidfVectorizer(tokenizer=lambda x: x.split('|')), 'genre'),
    ('year', 'passthrough', ['year']),
    ('encoded', 'passthrough', ['star', 'director', 'name'])
])

model = Pipeline(steps=[
    ('preprocessing', full_preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X_temp, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.3f}")
print(f"R² Score: {r2:.3f}")

sample_input = {
    'star': 'Tom Hanks',
    'name': 'The Time Traveler',
    'year': 2015,
    'genre': 'Drama|Sci-Fi|Romance',
    'director': 'Christopher Nolan'
}
input_df = pd.DataFrame([sample_input])

input_df['star'] = input_df['star'].fillna('Unknown_Star')
input_df['name'] = input_df['name'].fillna('Unknown_Name')
input_df['year'] = input_df['year'].fillna(df['year'].median())
input_df['genre'] = input_df['genre'].fillna('Unknown_Genre')
input_df['director'] = input_df['director'].fillna('Unknown_Director')

encoded_input = input_df.copy()
encoded_input[['star', 'director', 'name']] = target_enc.transform(
    input_df[['star', 'director', 'name']]
)

predicted_score = model.predict(encoded_input)[0]
print(f"Predicted Movie Score: {predicted_score:.2f}")

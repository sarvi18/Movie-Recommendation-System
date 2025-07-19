import pandas as pd
import nltk
import os
import shutil
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report

import nltk
nltk.download("punkt")

nltk.download('punkt_tab')

def nltk_tokenizer(text):
    return word_tokenize(text.lower())

df = pd.read_csv("netflix_titles.csv")
df = df.dropna(subset=['listed_in', 'description', 'director', 'cast', 'country', 'duration']).copy()
df['genres'] = df['listed_in'].str.split(',').apply(lambda x: [i.strip() for i in x])

df['combined_text'] = (
    df['description'] + ' ' +
    df['director'] + ' ' +
    df['cast'] + ' ' +
    df['country'] + ' ' +
    df['duration']
)

vectorizer = TfidfVectorizer(tokenizer=nltk_tokenizer, token_pattern=None, max_features=5000)
X_text = vectorizer.fit_transform(df['combined_text'])

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['genres'])

X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)

classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000))
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, target_names=mlb.classes_))

def predict_genres(description, director, cast, country, duration, threshold=0.23):
    combined = f"{description} {director} {cast} {country} {duration}"
    X_new = vectorizer.transform([combined])
    probabilities = classifier.predict_proba(X_new)[0]
    results = []
    for genre, prob in zip(mlb.classes_, probabilities):
        if prob >= threshold:
            results.append((genre, float(round(prob, 2))))
    results.sort(key=lambda x: x[1], reverse=True)

    if results:
        return [f"{genre} ({score * 100:.0f}%)" for genre, score in results]
    else:
        return ["No genres predicted above threshold"]


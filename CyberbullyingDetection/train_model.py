# ==========================================
# CYBERBULLYING DETECTION SYSTEM
# TF-IDF + Ensemble Model
# ==========================================

import pandas as pd
import re
import nltk
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ==========================================
# DOWNLOAD NLTK RESOURCES
# ==========================================

nltk.download("stopwords")
nltk.download("wordnet")


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("cyberbullying_tweets.csv")

print("Dataset Shape:", df.shape)
print(df.head())


# ==========================================
# TEXT PREPROCESSING
# ==========================================

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):

    text = str(text).lower()

    # remove links, mentions, symbols
    text = re.sub(r"http\S+|@\w+|[^a-zA-Z\s]", "", text)

    words = text.split()

    # remove stopwords
    words = [w for w in words if w not in stop_words]

    # lemmatization
    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)


df["clean_text"] = df["tweet_text"].apply(preprocess)


# ==========================================
# LABEL ENCODING
# ==========================================

le = LabelEncoder()
df["label"] = le.fit_transform(df["cyberbullying_type"])

print("\nClasses:", le.classes_)


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["label"],
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)


# ==========================================
# TF-IDF FEATURE EXTRACTION
# ==========================================

tfidf = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1,2),
    sublinear_tf=True
)

X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

print("\nTF-IDF shape:", X_train_vec.shape)


# ==========================================
# MODEL DEFINITIONS
# ==========================================

svm = LinearSVC(
    C=2,
    class_weight="balanced"
)

log_reg = LogisticRegression(
    max_iter=2000,
    n_jobs=-1
)

xgb = XGBClassifier(
    n_estimators=120,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist",
    eval_metric="mlogloss"
)


# ==========================================
# ENSEMBLE MODEL
# ==========================================

model = VotingClassifier(

    estimators=[
        ("svm", svm),
        ("lr", log_reg),
        ("xgb", xgb)
    ],

    voting="hard"

)


print("\nTraining model...")

model.fit(X_train_vec, y_train)


# ==========================================
# PREDICTION
# ==========================================

pred = model.predict(X_test_vec)


# ==========================================
# EVALUATION
# ==========================================

print("\nModel Accuracy:", accuracy_score(y_test, pred))

print("\nClassification Report:\n")
print(classification_report(y_test, pred))


# ==========================================
# TEST SAMPLE
# ==========================================

sample = ["You are a stupid idiot"]

sample_clean = [preprocess(sample[0])]

sample_vec = tfidf.transform(sample_clean)

prediction = model.predict(sample_vec)

print("\nSample Input:", sample[0])
print("Predicted Class:", le.inverse_transform(prediction))


# ==========================================
# SAVE MODEL FOR FLASK APP
# ==========================================

joblib.dump(model, "bullying_model.pkl")
joblib.dump(tfidf, "vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\nModel files saved successfully!")
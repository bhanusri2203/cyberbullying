import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier
import joblib


# -----------------------------
# 1. Load Dataset
# -----------------------------

data = pd.read_csv("cyberbullying_dataset.csv")

print("Dataset columns:", data.columns)


# Try common column names automatically
text_column = None
label_column = None

for col in data.columns:
    if col.lower() in ["text", "comment", "message", "tweet"]:
        text_column = col

    if col.lower() in ["label", "class", "target", "cyberbullying"]:
        label_column = col


if text_column is None or label_column is None:
    raise Exception("Dataset must contain a text column and label column")


X = data[text_column].astype(str)
y = data[label_column]


# -----------------------------
# 2. Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# 3. TF-IDF Vectorization
# -----------------------------

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=8000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# -----------------------------
# 4. PCA Dimensionality Reduction
# -----------------------------

pca = PCA(n_components=300)

X_train_pca = pca.fit_transform(X_train_vec.toarray())
X_test_pca = pca.transform(X_test_vec.toarray())


# -----------------------------
# 5. Train XGBoost Model
# -----------------------------

model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.08,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train_pca, y_train)


# -----------------------------
# 6. Predictions
# -----------------------------

predictions = model.predict(X_test_pca)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))


# -----------------------------
# 7. Save Model
# -----------------------------

joblib.dump(model, "bullying_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(pca, "pca_transform.pkl")

print("\nModel files saved successfully!")
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

MODEL_PATH = "model.joblib"

def train_model_from_csv(csv_path, save_path=MODEL_PATH):
    """Train the field classification model from labeled CSV"""
    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("Model evaluation:\n", classification_report(y_test, y_pred))

    joblib.dump(pipeline, save_path)
    print(f"Model saved to {save_path}")


def load_model(path=MODEL_PATH):
    """Load the trained model"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file '{path}' not found. Please train the model first.")
    return joblib.load(path)


def predict_field(text, model=None):
    """Predict the field label for a given text string"""
    if model is None:
        model = load_model()
    return model.predict([text])[0]

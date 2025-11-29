# src/predict.py
import os
import sys
import pickle

# Ensure preprocess imports correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import clean_text, combine_text


# File paths for model artifacts
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
VEC_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")


# Load model ONCE (fast)
_model = None
_vectorizer = None

def load():
    global _model, _vectorizer

    if _model is not None and _vectorizer is not None:
        return _model, _vectorizer

    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)

    with open(VEC_PATH, "rb") as f:
        _vectorizer = pickle.load(f)

    return _model, _vectorizer



# PREDICT FUNCTION
def predict_from_fields(subject, description, product=None):
    """
    Predict ticket type from user-entered fields.
    Returns: (prediction, probability)
    """
    model, vec = load()

    # Combine → Clean → Vectorize
    combined = combine_text(subject, description, product)
    cleaned = clean_text(combined)

    # If cleaned text is empty → cannot classify
    if cleaned.strip() == "":
        return "Unknown", 0.0

    X = vec.transform([cleaned])
    pred = model.predict(X)[0]

    # Probability
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X).max())

    return pred, proba



# CLI TESTING (useful for debugging)
if __name__ == "__main__":
    subj = input("Subject: ")
    desc = input("Description: ")

    pred, proba = predict_from_fields(subj, desc)

    print("\nPredicted Ticket Type:", pred)
    if proba is not None:
        print("Confidence:", round(proba, 2))

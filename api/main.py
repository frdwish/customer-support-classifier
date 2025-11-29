# api/main.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Import preprocessing functions
from src.preprocess import clean_text, combine_text

app = FastAPI(title="Customer Support Ticket Classifier API")


# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
VEC_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")


# Loading model once at startup

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VEC_PATH, "rb") as f:
        vectorizer = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model or vectorizer: {e}")


# Request model
class TicketIn(BaseModel):
    ticket_subject: str = ""
    ticket_description: str = ""
    product_purchased: str | None = None


# Predict endpoint
@app.post("/predict")
def predict(ticket: TicketIn):
    # Combine and clean text
    combined = combine_text(
        ticket.ticket_subject,
        ticket.ticket_description,
        ticket.product_purchased
    )
    cleaned = clean_text(combined)

    if cleaned.strip() == "":
        return {"ticket_type": "Unknown", "confidence": 0.0}

    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]

    result = {"ticket_type": str(pred)}

    # Confidence score if supported
    if hasattr(model, "predict_proba"):
        result["confidence"] = float(model.predict_proba(X).max())

    return result



# Root route
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Ticket classifier API is running"
    }

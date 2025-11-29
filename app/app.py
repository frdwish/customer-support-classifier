import streamlit as st
import pickle
import os
import matplotlib.pyplot as plt


# Load Model & Vectorizer Once

MODEL_PATH = os.path.join("models", "model.pkl")
VEC_PATH = os.path.join("models", "vectorizer.pkl")

@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VEC_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_artifacts()


# Combine & Clean Text

def clean_text(text):
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

    stop_words = set(stopwords.words("english"))
    lemm = WordNetLemmatizer()

    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemm.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def combine_text(subject, description, product):
    parts = []
    if subject:
        parts.append(subject)
    if description:
        parts.append(description)
    if product:
        parts.append(product)
    return " | ".join(parts)


# Streamlit UI

st.set_page_config(page_title="Ticket Classifier", layout="wide", page_icon="🎫")
st.title("🎫 Customer Support Ticket Classifier")
st.markdown("Predict the category of customer support tickets using Machine Learning.")


# Model Accuracy Display

st.subheader("Model Accuracy")
st.metric(label="Accuracy", value="0.85")


# User Inputs

st.subheader("Predict a Single Ticket")

col1, col2 = st.columns(2)
with col1:
    subject = st.text_input("Ticket Subject")
with col2:
    product = st.text_input("Product Purchased (optional)")

description = st.text_area("Ticket Description", height=150)


# Predict Button

if st.button("Predict Ticket Type"):
    if not subject and not description:
        st.error("Please enter at least subject or description.")
    else:
        text = combine_text(subject, description, product)
        clean = clean_text(text)
        X = vectorizer.transform([clean])
        pred = model.predict(X)[0]

        st.success(f"Predicted Ticket Type: **{pred}**")

        # Plot confidence if available
        if hasattr(model, "predict_proba"):
            confidence = model.predict_proba(X).max()
            fig, ax = plt.subplots()
            ax.bar(["Confidence"], [confidence])
            ax.set_ylim([0, 1])
            st.pyplot(fig)

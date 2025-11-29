import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/predict"

# --- Page Config ---
st.set_page_config(
    page_title="Ticket Classifier",
    layout="wide",
    page_icon="🎫"
)

# --- Load Custom CSS ---
try:
    with open("app/assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("CSS file not found. Continuing without custom style.")

# --- Title ---
st.title("🎫 Customer Support Ticket Classifier")
st.markdown("A production-grade NLP model to classify customer support tickets.")

# --- Session State for Persistent Inputs ---
if "subject" not in st.session_state:
    st.session_state.subject = ""
if "description" not in st.session_state:
    st.session_state.description = ""
if "product" not in st.session_state:
    st.session_state.product = ""

# --- Input Fields ---
col1, col2 = st.columns(2)
with col1:
    subject = st.text_input("Ticket Subject", value=st.session_state.subject)
with col2:
    product = st.text_input("Product Purchased (optional)", value=st.session_state.product)

description = st.text_area("Ticket Description", height=150, value=st.session_state.description)

# Update session state
st.session_state.subject = subject
st.session_state.description = description
st.session_state.product = product

# --- Predict Button ---
if st.button("Predict Ticket Type"):
    if not subject and not description:
        st.error("⚠ Please enter at least a subject or description.")
        st.stop()

    with st.spinner("Analyzing ticket..."):
        payload = {
            "ticket_subject": subject,
            "ticket_description": description,
            "product_purchased": product or None
        }

        try:
            response = requests.post(API_URL, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.RequestException as e:
            st.error("Failed to reach API. Make sure FastAPI is running.")
            st.write(str(e))
            st.stop()

    # --- Display Results ---
    confidence = result.get("confidence", 0.0)
    predicted_type = result.get("ticket_type", "Unknown")

    col1, col2 = st.columns(2)
    col1.metric("Predicted Ticket Type", predicted_type)
    col2.metric("Model Confidence", f"{confidence:.2f}")

    # --- Small Bar Chart for Model Accuracy vs Confidence ---
    model_accuracy = 0.85  # replace with your model's actual accuracy
    st.bar_chart(pd.DataFrame({"Accuracy": [model_accuracy], "Confidence": [confidence]}))

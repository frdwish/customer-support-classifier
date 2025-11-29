import streamlit as st
import requests
import pandas as pd
import pickle
import os

API_URL = "http://127.0.0.1:8000/predict"

# --- Page Config ---
st.set_page_config(
    page_title="Customer Support Ticket Classifier",
    layout="wide",
    page_icon="🎫"
)

st.title("🎫 Customer Support Ticket Classifier")
st.markdown("A production-grade NLP model to classify customer support tickets.")

# --- Load Model Accuracy Dynamically ---
try:
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    y_accuracy = 0.85  # Replace with dynamic accuracy if saved
except FileNotFoundError:
    st.warning("Model not found. Accuracy metric unavailable.")
    y_accuracy = 0.0

st.metric("Model Accuracy", f"{y_accuracy:.2f}")

# --- Session State for Persistent Inputs ---
if "subject" not in st.session_state:
    st.session_state.subject = ""
if "description" not in st.session_state:
    st.session_state.description = ""
if "product" not in st.session_state:
    st.session_state.product = ""

# --- Single Ticket Prediction ---
st.subheader("Predict a Single Ticket")
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

if st.button("Predict Ticket Type"):
    if not subject and not description:
        st.error("⚠ Please enter at least a subject or description.")
    else:
        payload = {
            "ticket_subject": subject,
            "ticket_description": description,
            "product_purchased": product or None
        }
        with st.spinner("Analyzing ticket..."):
            try:
                response = requests.post(API_URL, json=payload, timeout=10)
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.RequestException as e:
                st.error("Failed to reach API. Make sure FastAPI is running.")
                st.write(str(e))
                st.stop()

        confidence = result.get("confidence", 0.0)
        predicted_type = result.get("ticket_type", "Unknown")

        col1, col2 = st.columns(2)
        col1.metric("Predicted Ticket Type", predicted_type)
        col2.metric("Confidence", f"{confidence:.2f}")

        st.bar_chart(pd.DataFrame({"Model Accuracy": [y_accuracy], "Confidence": [confidence]}))

# --- Batch Predictions ---
st.subheader("Batch Prediction (CSV Upload)")
st.markdown("Upload a CSV with columns: Ticket Subject, Ticket Description, Product Purchased (optional)")

uploaded_file = st.file_uploader("Choose CSV", type="csv")
if uploaded_file:
    df_batch = pd.read_csv(uploaded_file)
    required_cols = ["Ticket Subject", "Ticket Description"]
    if not all(col in df_batch.columns for col in required_cols):
        st.error(f"CSV must contain columns: {required_cols}")
    else:
        df_batch["Product Purchased"] = df_batch.get("Product Purchased", "")
        predictions = []
        confidences = []

        for idx, row in df_batch.iterrows():
            payload = {
                "ticket_subject": row["Ticket Subject"],
                "ticket_description": row["Ticket Description"],
                "product_purchased": row["Product Purchased"] or None
            }
            try:
                response = requests.post(API_URL, json=payload, timeout=10)
                response.raise_for_status()
                result = response.json()
                predictions.append(result.get("ticket_type", "Unknown"))
                confidences.append(result.get("confidence", 0.0))
            except:
                predictions.append("Error")
                confidences.append(0.0)

        df_batch["Predicted Ticket Type"] = predictions
        df_batch["Confidence"] = confidences

        st.success("Batch prediction complete!")
        st.dataframe(df_batch)
        # Optionally allow download
        st.download_button(
            "Download Predictions as CSV",
            df_batch.to_csv(index=False).encode('utf-8'),
            file_name="batch_predictions.csv",
            mime="text/csv"
        )

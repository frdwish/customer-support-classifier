import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"
st.title("🧪 API Tester")
st.write("Send live requests to the FastAPI backend.")

subject = st.text_input("Subject")
description = st.text_area("Description")
product = st.text_input("Product (optional)")

if st.button("Test API"):
    payload = {
        "ticket_subject": subject,
        "ticket_description": description,
        "product_purchased": product or None
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        st.success(f"Predicted Ticket Type: **{result['ticket_type']}**")
        if "confidence" in result:
            st.write(f"Confidence: {result['confidence']:.2f}")
    except Exception as e:
        st.error("Failed to reach API. Make sure the Render API is live.")
        st.write(e)

import streamlit as st
from handlers import handle_upload, display_images, run_ocr_and_predict, feedback_section
from model import load_model
from handlers import download_structured_csv

st.set_page_config(page_title="AI_Invoicer", layout="centered")
st.title("🧾 AI Invoicer – Extract Data from Invoices")

uploaded_file = st.file_uploader("Upload an invoice (PDF or image)", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    images = handle_upload(uploaded_file)
    display_images(images)
    
    model = load_model()
    if not model:
        st.error("Model could not be loaded.")
    else:
        extracted_texts, predictions = run_ocr_and_predict(images, model)
        if predictions:
            st.subheader("📄 Download Results")
            download_structured_csv(predictions)
        feedback_section(extracted_texts)

st.markdown("📬 Developed by [Mohammad Bashar] — Powered by EasyOCR & Streamlit")

import fitz
import streamlit as st
import csv
import pandas as pd
from collections import defaultdict
from ocr_utils import prepare_image_for_ocr, extract_text_from_image
from model import predict_field

FEEDBACK_CSV = "feedback.csv"



def download_structured_csv(structured_data, filename="structured_invoice.csv"):
    rows = []
    for label, lines in structured_data.items():
        if label.lower() == "junk":
            continue
        for line in lines:
            rows.append({"Label": label, "Text": line})

    if not rows:
        st.info("No valid structured data to download.")
        return

    df = pd.DataFrame(rows)
    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Structured Results as CSV",
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        key="download-structured-csv_{filename}",
    )



def handle_upload(uploaded_file):
    file_bytes = uploaded_file.read()
    images = [] 
    if uploaded_file.type == "application/pdf":
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            pix = page.get_pixmap()
            images.append(pix.tobytes("png"))
    else:
        images = [file_bytes]
    return images


def display_images(images):
    for i, img in enumerate(images):
        st.subheader(f"🖼️ Page {i+1}")
        st.image(img, use_container_width=True)


def run_ocr_and_predict(images, model):
    extracted_texts = []
    structured_data = defaultdict(list)

    for i, img in enumerate(images):
        processed = prepare_image_for_ocr(img)
        if processed is not None:
            ocr_text = extract_text_from_image(img)
            extracted_texts.append(ocr_text)
            st.text_area("📄 Extracted Text", ocr_text, height=200, key=f"ocr_{i}")

            for line in ocr_text.splitlines():
                if line.strip():
                    label = predict_field(line, model)
                    structured_data[label].append(line)
        else:
            st.warning("⚠️ Could not process the image for OCR.")

    if structured_data:
        st.subheader("🤖 Field Detection (ML Model)")
        st.json({k: v for k, v in structured_data.items() if v})
    
    return extracted_texts, structured_data


def feedback_section(extracted_texts):
    st.subheader("✍️ Provide Feedback")

    for i, text in enumerate(extracted_texts):
        st.markdown(f"#### Feedback for Page {i+1}")
        lines = text.splitlines()

        if f'edited_lines_{i}' not in st.session_state:
            st.session_state[f'edited_lines_{i}'] = lines.copy()

        for line_idx, line in enumerate(lines):
            if line.strip():
                with st.expander(f"📝 Line {line_idx+1}"):
                    edited_text = st.text_input(
                        "Edit text",
                        value=st.session_state[f'edited_lines_{i}'][line_idx],
                        key=f"edit_{i}_{line_idx}"
                    )
                    st.session_state[f'edited_lines_{i}'][line_idx] = edited_text

                    label = st.text_input("Label for this line", key=f"label_{i}_{line_idx}")

                    if st.button("✅ Submit Feedback", key=f"submit_{i}_{line_idx}"):
                        with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow([edited_text, label])
                        st.success("Feedback submitted!")

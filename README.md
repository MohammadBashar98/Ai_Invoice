# Ai_Invoice
HOPn Task
# 🧾 AI_Invoicer

A smart OCR-powered invoice parser built using **Streamlit**, **EasyOCR**, and **PyMuPDF**. Upload image or PDF invoices and get structured field detection using an ML model.

## 🚀 Features
- PDF and image support (PNG, JPG, JPEG)
- OCR extraction (using EasyOCR)
- ML-based label prediction (custom model)
- Editable feedback section to collect user corrections
- Streamlit UI ready for deployment

## 🖼️ How It Works
1. Upload an invoice
2. Text is extracted using OCR
3. Each line is passed through a model to predict labels
4. Users can provide feedback per line

## ✍️ Feedback & Model Retraining

When users submit feedback by editing the extracted text and assigning a label, the results are saved to a local CSV file (`Book2.csv`). Each row contains:
- The corrected text
- The correct label (e.g., `InvoiceNumber`, `Date`, `Total`, or `junk`)

This CSV acts as a growing dataset of high-quality labeled examples.

### 🔁 Model Retraining
Currently, retraining the ML model is done **manually** using the accumulated feedback in the CSV. This allows the model to learn from user corrections and improve its predictions over time.

You can periodically trigger retraining using a custom script like:

```python
from train import train_model_from_csv
train_model_from_csv("feedback.csv")

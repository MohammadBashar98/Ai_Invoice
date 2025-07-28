import re

field_patterns = {
    'invoice_number': r'(invoice|inv|bill|document)\s*(no|num|number|#)?',
    'date': r'(date|issued|invoice\s*date)',
    'total': r'(total|amount|grand\s*total|balance)',
    'tax': r'(tax|vat|gst)',
    'description': r'(description|item|product|service)',
    'quantity': r'(qty|quantity)',
    'unit_price': r'(price|unit\s*price|rate)'
}

def detect_fields(columns, model=None):
    detected = {}
    for col in columns:
        text = str(col).lower()
        if model:
            predicted_field = model.predict([text])[0]
            detected[predicted_field] = col
        else:
            for field, pattern in field_patterns.items():
                if re.search(pattern, text):
                    detected[field] = col
                    break
    return detected
import cv2
import numpy as np
import easyocr

reader = easyocr.Reader(['en'], gpu=False)

def prepare_image_for_ocr(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return processed
    except Exception as e:
        return None

def extract_text_from_image(processed_image_bytes):
    try:
        nparr = np.frombuffer(processed_image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        result = reader.readtext(img, detail=0)
        return "\n".join(result).strip()
    except Exception as e:
        return None

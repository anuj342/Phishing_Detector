# app/app.py

import os
import re
import joblib
import numpy as np
# REMOVED: TensorFlow, cv2, and mtcnn are no longer needed
# import tensorflow as tf
# import cv2
# from mtcnn.mtcnn import MTCNN
import email
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
from flask import Flask, request, jsonify

# --- Initial Setup ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Define paths to saved models
PHISHING_MODEL_PATH = '../models/phishing_detector_model.joblib'
VECTORIZER_PATH = '../models/tfidf_vectorizer.joblib'
# REMOVED: Deepfake model path is no longer needed
# DEEPFAKE_MODEL_PATH = 'models/deepfake_detector_model.keras'

# --- Load Models at Startup ---
logging.info("Loading models, please wait...")
try:
    phishing_model = joblib.load(PHISHING_MODEL_PATH)
    tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
    # REMOVED: Loading of deepfake_model and face_detector
    logging.info("Models loaded successfully!")
except Exception as e:
    logging.error(f"Error loading models: {e}")
    exit()

# Define allowed file extensions
ALLOWED_EXTENSIONS_EMAIL = {'eml'}
# REMOVED: Video file extensions are no longer needed
# ALLOWED_EXTENSIONS_VIDEO = {'mp4', 'mov', 'avi'}

# --- Phishing Prediction Logic ---

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Cleans text by removing URLs, non-alphabetic characters, and stop words, and applies lemmatization."""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

suspicious_keywords = ['verify', 'account', 'password', 'urgent', 'suspend', 'confirm', 'login', 'secure', 'bank', 'credit']

def count_suspicious_keywords(text):
    """Counts the occurrences of suspicious keywords in the text."""
    count = 0
    for keyword in suspicious_keywords:
        if keyword in text:
            count += 1
    return count

def has_ip_in_url(text):
    """Checks if a URL in the text uses an IP address instead of a domain name."""
    ip_pattern = r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    return 1 if re.search(ip_pattern, text) else 0

def predict_phishing(email_content_bytes):
    """Analyzes email content to predict if it's phishing."""
    try:
        msg = email.message_from_bytes(email_content_bytes)
        email_body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    email_body = part.get_payload(decode=True).decode(errors='ignore')
                    break
        else:
            email_body = msg.get_payload(decode=True).decode(errors='ignore')

        if not email_body:
            return {"error": "Could not extract text body from email."}

        cleaned_body = clean_text(email_body)
        keyword_count = count_suspicious_keywords(cleaned_body)
        has_ip = has_ip_in_url(email_body)
        text_features = tfidf_vectorizer.transform([cleaned_body]).toarray()
        numerical_features = np.array([[keyword_count, has_ip]])
        all_features = np.hstack([numerical_features, text_features])

        prediction = phishing_model.predict(all_features)
        probability = phishing_model.predict_proba(all_features)
        result = "Phishing" if prediction[0] == 1 else "Legitimate"
        confidence = float(np.max(probability))

        return {
            "prediction": result,
            "confidence": f"{confidence:.2%}"
        }
    except Exception as e:
        logging.error(f"Error during phishing prediction: {e}")
        return {"error": "An internal error occurred during phishing analysis."}

# --- REMOVED: Deepfake Prediction Logic ---
# The entire predict_deepfake function has been removed.

# --- Flask API Endpoint ---

def allowed_file(filename, allowed_set):
    """Checks if the file extension is in the allowed set."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to receive a file and return a prediction."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = file.filename
    # The endpoint now only checks for email files.
    if allowed_file(filename, ALLOWED_EXTENSIONS_EMAIL):
        email_bytes = file.read()
        result = predict_phishing(email_bytes)
        return jsonify(result)
    # REMOVED: The elif block for handling videos has been removed.
    else:
        return jsonify({"error": "Unsupported file type. Please upload a .eml file."}), 400

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

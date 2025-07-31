# app/app.py

import re
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Initial Setup ---
app = Flask(__name__)

# Define paths to the saved phishing model and vectorizer
PHISHING_MODEL_PATH = '../models/phishing_detector_model.joblib'
VECTORIZER_PATH = '../models/tfidf_vectorizer.joblib'

# --- Load Model at Startup ---
print("Loading phishing model, please wait...")
phishing_model = joblib.load(PHISHING_MODEL_PATH)
tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
print("Model loaded successfully!")

# --- Phishing Prediction Logic ---

# Re-create the text cleaning function from Phase 1
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Re-create the feature engineering functions from Phase 1
suspicious_keywords = ['verify', 'account', 'password', 'urgent', 'suspend', 'confirm', 'login', 'secure']

def count_suspicious_keywords(text):
    count = 0
    for keyword in suspicious_keywords:
        if keyword in text:
            count += 1
    return count

def has_ip_in_url(text):
    ip_pattern = r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    return 1 if re.search(ip_pattern, text) else 0

def predict_phishing(email_body):
    if not email_body or not isinstance(email_body, str):
        return {"error": "Invalid input: Email content is missing or not text."}

    # 1. Clean the text
    cleaned_body = clean_text(email_body)

    # 2. Feature Engineering
    keyword_count = count_suspicious_keywords(cleaned_body)
    has_ip = has_ip_in_url(email_body) # Check original body for URLs
    
    # Vectorize text using the loaded vectorizer
    text_features = tfidf_vectorizer.transform([cleaned_body]).toarray()
    
    # Combine all features in the correct order
    numerical_features = np.array([[keyword_count, has_ip]])
    all_features = np.hstack([numerical_features, text_features])

    # 3. Make Prediction
    prediction = phishing_model.predict(all_features)
    probability = phishing_model.predict_proba(all_features)

    result = "Phishing" if prediction == 1 else "Legitimate"
    confidence = float(np.max(probability))

    return {
        "prediction": result,
        "confidence": f"{confidence:.2%}"
    }

# --- Flask API Endpoints ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods= ['POST'] )
def predict():
    try:
        data = request.get_json()
        if not data or 'email_text' not in data:
            return jsonify({"error": "Invalid request: Missing 'email_text' field."}), 400
        
        email_text = data['email_text']
        result = predict_phishing(email_text)
        return jsonify(result)

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)
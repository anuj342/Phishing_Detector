import os
import re
import joblib
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
import cv2
from mtcnn.mtcnn import MTCNN
import email
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#Initial setup
app = Flask(__name__)

#Define paths to saved models
PHISHING_MODEL_PATH= '../models/phshing_detector_model.joblib'
VECTORIZER_PATH = '../models/tfidf_vectorizer.joblib'
DEEPFAKE_MODEL_PATH = '../models/deepfake_detector_model.keras'

# --- Load Models at Startup ---
print("Loading models, please wait...")
phishing_model = joblib.load(PHISHING_MODEL_PATH)
tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
deepfake_model = tf.keras.models.load_model(DEEPFAKE_MODEL_PATH)
face_detector = MTCNN()
print("Models loaded successfully!")

# Define allowed file extensions
ALLOWED_EXTENSIONS_EMAIL = {'eml'}
ALLOWED_EXTENSIONS_VIDEO = {'mp4', 'mov', 'avi'}

# --- Load Models at Startup ---
print("Loading models, please wait...")
phishing_model = joblib.load(PHISHING_MODEL_PATH)
tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
deepfake_model = tf.keras.models.load_model(DEEPFAKE_MODEL_PATH)
face_detector = MTCNN()
print("Models loaded successfully!")

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

def predict_phishing(email_content_bytes):
    # Parse email content from bytes
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
    prediction = phishing_model.predict(all_features)[0]
    probability = phishing_model.predict_proba(all_features)[0]

    result = "Phishing" if prediction == 1 else "Legitimate"
    confidence = float(probability[prediction])

    return {
        "prediction": result,
        "confidence": f"{confidence:.2%}"
    }

# --- Deepfake Prediction Logic ---
def predict_deepfake(video_file_storage):
    # Save the video file temporarily to process it with OpenCV
    temp_video_path = "temp_video.mp4"
    video_file_storage.save(temp_video_path)

    cap = cv2.VideoCapture(temp_video_path)
    frame_predictions = []
    frame_count = 0

    while cap.isOpened() and frame_count < 90: # Analyze up to 90 frames (3 seconds at 30fps)
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 15th frame to speed up analysis
        if frame_count % 15 == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.detect_faces(frame_rgb)

            if results:
                # Use the first detected face
                x1, y1, width, height = results[0]['box']
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = x1 + width, y1 + height
                face = frame_rgb[y1:y2, x1:x2]

                if face.size != 0:
                    # Preprocess face for the model
                    face_resized = cv2.resize(face, (128, 128))
                    face_array = np.expand_dims(face_resized, axis=0) / 255.0

                    # Make prediction on the single face
                    pred = deepfake_model.predict(face_array)[0][0]
                    frame_predictions.append(pred)

        frame_count += 1

    cap.release()
    os.remove(temp_video_path) # Clean up the temporary file

    if not frame_predictions:
        return {"prediction": "Inconclusive", "confidence": "0.00%", "reason": "No faces detected in the video."}

    # Aggregate results: if average prediction > 0.5, it's likely fake
    avg_prediction = np.mean(frame_predictions)
    result = "Deepfake" if avg_prediction > 0.5 else "Real"
    confidence = avg_prediction if result == "Deepfake" else 1 - avg_prediction

    return {
        "prediction": result,
        "confidence": f"{confidence:.2%}"
    }

# --- Flask API Endpoint ---
def allowed_file(filename, allowed_set):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = file.filename
    if allowed_file(filename, ALLOWED_EXTENSIONS_EMAIL):
        # It's an email file, process for phishing
        email_bytes = file.read()
        result = predict_phishing(email_bytes)
        return jsonify(result)

    elif allowed_file(filename, ALLOWED_EXTENSIONS_VIDEO):
        # It's a video file, process for deepfake
        result = predict_deepfake(file)
        return jsonify(result)

    else:
        return jsonify({"error": "Unsupported file type. Please upload a .eml, .mp4, .mov, or .avi file."}), 400

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

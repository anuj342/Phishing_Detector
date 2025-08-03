#AI-Powered Phishing Detector
A web-based application that uses machine learning and natural language processing to analyze email content in real-time and determine its likelihood of being a phishing attempt.

##Table of Contents
The Problem

The Solution

Live Demo Preview

Tech Stack & Architecture

Project Structure

Installation & Usage

Model Training & Evaluation

Future Work

License

The Problem
In today's digital landscape, phishing remains one of the most pervasive and damaging cyber threats. Traditional signature-based filters struggle to keep up with the increasing sophistication of phishing campaigns, which now often employ social engineering tactics, urgent language, and deceptive URLs to trick users. This project tackles this challenge by moving beyond simple keyword matching to a more intelligent, feature-based detection approach powered by machine learning.

The Solution
This application provides a simple yet powerful interface for users to instantly analyze suspicious emails. By pasting the full content of an email, the tool performs a multi-faceted analysis in seconds:

Key Features:

Intelligent Text Analysis: Leverages a TF-IDF Vectorizer and a trained Random Forest model to understand the context and nuance of the email text, identifying patterns common in phishing attacks.

URL-Based Feature Engineering: Automatically extracts and analyzes URLs within the email for suspicious characteristics, such as the presence of IP addresses, @ symbols, or an unusual number of subdomains.

Content-Based Heuristics: Scans for high-risk keywords (e.g., "urgent," "verify," "password") that are frequently used in phishing campaigns.

Real-Time Risk Scoring: Provides an immediate, easy-to-understand prediction ("Phishing" or "Legitimate") along with a confidence score.

Live Demo Preview
Here's a look at the user interface and the analysis result:

A user pastes a suspicious email into the text area.

The model correctly identifies the email as a phishing attempt with a high confidence score.

Tech Stack & Architecture
This project integrates a machine learning backend with a web frontend, following a simple client-server architecture.

Backend:

Language: Python 3.9

Web Framework: Flask

ML Library: Scikit-learn

NLP & Data: Pandas, NLTK, Joblib

Frontend:

Structure: HTML5

Styling: CSS3

Dynamic Behavior: JavaScript (with Fetch API)

Architecture Diagram:

graph TD
    A[User's Browser <br> (index.html, script.js)] -- 1. Paste Email & Submit --> B{Flask Backend <br> (app.py)};
    B -- 2. Preprocess Text --> C[Feature Engineering <br> (Text Cleaning, URL & Content Features)];
    C -- 3. Create Feature Vector --> D[TF-IDF Vectorizer <br> (tfidf_vectorizer.pkl)];
    D -- 4. Feed Features --> E[Phishing Model <br> (phishing_model.pkl)];
    E -- 5. Make Prediction --> B;
    B -- 6. Return JSON Response <br> (Prediction & Confidence) --> A;
    A -- 7. Display Result --> F[Result on Page];

Project Structure
The project is organized into a modular structure for clarity and scalability.

phishing-detector/
│
├── app/
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css
│   │   └── js/
│   │       └── script.js
│   ├── templates/
│   │   └── index.html
│   └── app.py
│
├── models/
│   ├── phishing_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   └── Phishing_Model_Training.ipynb
│
├── datasets/
│   └── (Your training CSV/data files)
│
├── requirements.txt
└── README.md

Installation & Usage
To run this project locally, follow these steps.

Prerequisites:

Python 3.9 or later

pip package manager

A virtual environment (recommended)

Step 1: Clone the Repository

git clone [https://github.com/YOUR_USERNAME/phishing-detector.git](https://github.com/YOUR_USERNAME/phishing-detector.git)
cd phishing-detector

Step 2: Create and Activate a Virtual Environment

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate

Step 3: Install Dependencies

pip install -r requirements.txt

Note: You will also need to download NLTK data. The app.py script handles this automatically on the first run.

Step 4: Ensure Models are in Place
This project assumes you have already trained the model and vectorizer. Make sure phishing_model.pkl and tfidf_vectorizer.pkl are located inside the models/ directory. If not, you must run the training notebook first.

Step 5: Run the Flask Application

python app/app.py

Step 6: Access the Application
Open your web browser and navigate to http://127.0.0.1:5000. You can now paste email content into the text box and click "Analyze" to see the prediction.

Model Training & Evaluation
The phishing classification model was trained using a dataset of legitimate and phishing emails.

Model: Random Forest Classifier

Feature Extraction:

TF-IDF on cleaned email body text.

Custom features: URL length, domain length, number of subdomains, presence of IP/@ symbol.

Keyword counters for words like "urgent", "verify", etc.

Performance: The model achieved XX.X% accuracy on the held-out test set, with a strong F1-score demonstrating a good balance between precision and recall.

For a complete breakdown of the data cleaning, feature engineering, and model training process, please see the Jupyter Notebook: notebooks/Phishing_Model_Training.ipynb.

Future Work
This project provides a solid foundation that can be extended in several ways:

Threat Intelligence Integration: Integrate a real-time API like VirusTotal to check the reputation of URLs and IPs found in the email.

Header Analysis: Expand the feature set to include email header analysis (e.g., checking SPF, DKIM, DMARC records) for more robust spoofing detection.

Browser Extension: Develop a browser extension that allows users to analyze emails directly within their webmail client (e.g., Gmail, Outlook).

Advanced Models: Experiment with more advanced deep learning models like LSTMs or Transformers for text classification to potentially improve accuracy.

License
This project is licensed under the MIT License. See the LICENSE file for details.

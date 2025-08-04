# AI-Powered Phishing Email Detector
A web application that uses a machine learning model and Natural Language Processing (NLP) to analyze raw email text and classify it as either legitimate or a phishing attempt in real-time.

## Table of Contents
Problem Statement

Features

Tech Stack

How It Works

Setup and Installation

Usage

Project Structure

Future Improvements

License

## Problem Statement
Phishing is one of the most pervasive and damaging cyber threats, responsible for data breaches, financial loss, and credential theft. Traditional email filters are effective but not foolproof, and attackers are constantly evolving their techniques. This project provides an accessible tool for users to get an instant second opinion on a suspicious email, empowering them to make safer decisions.

## Features
Real-Time Analysis: Instantly classifies email content pasted into the web interface.

Confidence Scoring: Provides a probability score to indicate the model's confidence in its prediction.

NLP-Powered Feature Engineering: Goes beyond simple keyword matching by analyzing text structure, URL patterns, and other linguistic features.

Simple & Intuitive UI: A clean and straightforward user interface built with HTML, CSS, and JavaScript.

RESTful API: A backend built with Flask that serves the machine learning model.

## Tech Stack
Backend: Python, Flask

Machine Learning: Scikit-learn, Pandas, NumPy

NLP: NLTK (Natural Language Toolkit)

Frontend: HTML5, CSS3, JavaScript (Fetch API)

## How It Works
The application follows a simple but powerful workflow to analyze email content:

User Input: The user pastes the full raw content of an email into the text area on the web page.

API Request: The frontend JavaScript captures the text and sends it to the Flask backend API (/predict) via an asynchronous fetch request.

Backend Processing: The Flask server receives the text data.

Feature Engineering: A dedicated Python function extracts meaningful features from the raw text. This includes:

Parsing the email body and headers.

Extracting and analyzing URLs (e.g., checking for IP addresses, multiple subdomains).

Calculating the frequency of suspicious keywords (e.g., "verify", "urgent", "password").

Prediction: The engineered features are fed into the pre-trained Random Forest model.

API Response: The model returns a prediction ("Phishing" or "Legitimate") and a confidence score. The backend packages this into a JSON response.

Display Result: The frontend JavaScript parses the JSON response and dynamically updates the UI to display the final verdict to the user.

## Setup and Installation
To run this project locally, please follow these steps:

Clone the Repository

Bash

git clone https://github.com/YOUR_USERNAME/phishing-detector.git
cd phishing-detector
Create and Activate a Virtual Environment

Bash

## For macOS/Linux
python3 -m venv venv
source venv/bin/activate

## For Windows
python -m venv venv
venv\Scripts\activate
Install Dependencies
First, ensure you have a requirements.txt file. You can create one by running:

Bash

pip freeze > requirements.txt
Then, install all required packages:

Bash

pip install -r requirements.txt
Download NLTK Data
You need to download the necessary NLTK corpora. Run a Python interpreter and enter the following:

Python

import nltk
nltk.download('punkt')
nltk.download('stopwords')
Run the Application

Bash

cd app
python app.py
The application will be available at http://127.0.0.1:5000.

## Usage
Open your web browser and navigate to http://127.0.0.1:5000.

Copy the full content of an email you wish to analyze.

Paste the content into the text area provided.

Click the "Analyze Email" button.

The prediction and confidence score will appear below the button.

## Project Structure
phishing-detector/
├── app/
│   ├── static/
│   │   ├── style.css       # CSS for styling
│   │   └── script.js       # JavaScript for frontend logic
│   ├── templates/
│   │   └── index.html      # Main HTML page
│   └── app.py              # Flask application
├── assets/
│   └── screenshot.png      # Screenshot of the application
├── models/
│   ├── phishing_model.joblib # The trained ML model
│   └── tfidf_vectorizer.joblib # The saved TF-IDF vectorizer
├── .gitignore
├── README.md
└── requirements.txt
## Future Improvements
Threat Intelligence Integration: Integrate with APIs from services like VirusTotal or PhishTank to check the reputation of URLs found in emails in real-time.

Containerization: Package the application using Docker for easier deployment and scalability.

Build a Browser Extension: Create a browser extension (e.g., for Chrome/Firefox) that allows users to analyze emails directly from their webmail client (like Gmail) with a single click.

Advanced Models: Experiment with more advanced deep learning models like LSTMs or Transformers (e.g., BERT) for text classification to potentially improve accuracy.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

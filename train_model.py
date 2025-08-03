import pandas as pd
import re
import os
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# error will mostly occur, in that case uncomment the below 2 line (download just once)
# nltk.download('stopwords')
# nltk.download('wordnet')

# load data
print("Step 1: Loading data...")
try:
    # Make sure your dataset has exact label as 'body' and 'label'
    df = pd.read_csv('datasets/phishing.csv')
except FileNotFoundError:
    print("Error: 'datasets/phishing.csv' not found. Please check the filename and location.")
    exit()

print("Initial data head:")
print(df.head())
print("-" * 30)


# Preprocess data (important)
print("Step 2: Preprocessing data...")
df.dropna(subset=['body', 'label'], inplace=True)

df['label'] = pd.to_numeric(df['label'], errors='coerce')
df.dropna(subset=['label'], inplace=True) # Drop rows where conversion failed
df['label'] = df['label'].astype(int)

# This diagnostic print will show you the unique labels AFTER conversion.
# You should see both [0 1] or similar in the output.
print("Unique labels in the dataset:", df['label'].unique())
print("Label counts after standardization (check for both 0s and 1s):")
print(df['label'].value_counts())
print("-" * 30)


# Cleaning the data before use
print("Step 3: Cleaning email text...")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Apply cleaning to the 'body' column
df['body'] = df['body'].apply(clean_text)
print("Data cleaning complete.")
print("-" * 30)


# Feature engineering
print("Step 4: Performing feature engineering...")
suspicious_keywords = ['verify', 'account', 'password', 'urgent', 'suspend', 'confirm', 'login', 'secure']

def count_suspicious_keywords(text):
    return sum(1 for keyword in suspicious_keywords if keyword in text)

def has_ip_in_url(text):
    ip_pattern = r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    return 1 if re.search(ip_pattern, text) else 0

df['suspicious_keyword_count'] = df['body'].apply(count_suspicious_keywords)
df['has_ip_url'] = df['body'].apply(has_ip_in_url)

vectorizer = TfidfVectorizer(max_features=2000)
tfidf_features = vectorizer.fit_transform(df['body']).toarray()
tfidf_df = pd.DataFrame(tfidf_features, columns=vectorizer.get_feature_names_out())

df.reset_index(drop=True, inplace=True)
tfidf_df.reset_index(drop=True, inplace=True)

final_features_df = pd.concat([df[['suspicious_keyword_count', 'has_ip_url']], tfidf_df], axis=1)

X = final_features_df
y = df['label'] # Using the lowercase 'label' column
print("Feature engineering complete.")
print("Shape of final features (X):", X.shape)
print("-" * 30)


# Training model
print("Step 5: Splitting data and training the model...")
# Check if there's more than one class before stratifying
if len(y.unique()) > 1:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
else:
    # If only one class, stratification is not possible
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Warning: Only one class found in the dataset. Stratification is disabled.")


rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_classifier.fit(X_train, y_train)
print("Model training complete.")
print("-" * 30)


# Checking the accuracy
print("Step 6: Evaluating and saving the model...")
y_pred = rf_classifier.predict(X_test)

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(rf_classifier, 'models/phishing_detector_model.joblib')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
print("\nModel and vectorizer have been saved successfully to the 'models' folder.")

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- (You only need to run the downloads once) ---
#nltk.download('stopwords')
#nltk.download('wordnet')
# --------------------------------------------------

# 1. Load the Data
print("Loading data...")
df = pd.read_csv('datasets/phishing.csv')
print("Initial data head:")
print(df.head())
print("-" * 30)

# 2. Handle Missing Values & Standardize Labels
print("Preprocessing data...")
# Using the correct column names: 'Body' and 'Label'
df.dropna(subset=['body', 'label'], inplace=True)

# Using the correct column name: 'Label'
df['label'] = df['label'].apply(lambda x: 1 if 'phishing' in str(x).lower() else 0)
print("Label counts after standardization:")
print(df['label'].value_counts())
print("-" * 30)

# 3. Clean the Email Text
print("Cleaning email text...")
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

# Using the correct column name: 'Body'
df['body'] = df['body'].apply(clean_text)
print("Cleaned data head:")
print(df.head())
print("-" * 30)

print("Data preprocessing complete!")
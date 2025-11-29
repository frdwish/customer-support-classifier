# src/train.py
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.preprocess import clean_text, combine_text

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)


# 1. Load Cleaned Dataset
df = pd.read_csv("data/clean_tickets.csv", encoding="ISO-8859-1", engine="python", on_bad_lines="skip")
df.columns = df.columns.str.strip()

# Combine subject + description + product
df['text'] = df.apply(lambda row: combine_text(
    row['Ticket Subject'], 
    row['Ticket Description'], 
    row.get('Product Purchased')
), axis=1)

# Clean text
df['clean_text'] = df['text'].apply(clean_text)

# Drop rows with empty text or missing labels
df = df[df['clean_text'].str.strip() != ""]
df = df[df['Ticket Type'].notnull()]


# 2. Prepare Features & Labels
X = df['clean_text']
y = df['Ticket Type']

# TF-IDF vectorization
tfidf = TfidfVectorizer(
    ngram_range=(1,3),  # unigrams, bigrams, trigrams
    max_features=8000,
    min_df=2,
    max_df=0.9
)
X_vec = tfidf.fit_transform(X)

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)


# 3. Train Model
model = RandomForestClassifier(
    n_estimators=500,
    class_weight='balanced',  # handle label imbalance
    random_state=42
)
model.fit(X_train, y_train)


# 4. Evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# 5. Save Model & Vectorizer
pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(tfidf, open("models/vectorizer.pkl", "wb"))
print("✅ Model and vectorizer saved successfully!")

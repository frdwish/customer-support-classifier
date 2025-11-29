# src/preprocess.py

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download required NLTK resources once

try:
    stop_words = set(stopwords.words("english"))
except:
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    stop_words = set(stopwords.words("english"))

# not removing these words
critical_negations = {"no", "not", "never"}

stop_words = set([w for w in stop_words if w not in critical_negations])

lemm = WordNetLemmatizer()



# Text Cleaning
def clean_text(text: str) -> str:
    """Clean text: lowercase, remove URLs/special chars, remove stopwords, lemmatize."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    tokens = text.split()

    # keep negation words because they change meaning
    tokens = [t for t in tokens if t not in stop_words]

    tokens = [lemm.lemmatize(t) for t in tokens]

    return " ".join(tokens)



# combine text for description
# (must match training structure)
def combine_text(subject: str, description: str, product: str = None) -> str:
    """Combine subject + description + optional product for prediction."""
    parts = []

    if isinstance(subject, str) and subject.strip():
        parts.append(subject.strip())

    if isinstance(description, str) and description.strip():
        parts.append(description.strip())

    if product and isinstance(product, str) and product.strip():
        parts.append(product.strip())

    # During training you used: subject + " " + description
    return " ".join(parts)

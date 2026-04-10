"""
preprocess.py
-------------
Text cleaning functions for the hate speech dataset.
"""

import re
import nltk
from nltk.corpus import stopwords

# Download stopwords on first run
nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Clean a raw tweet string.

    Steps:
        1. Lowercase
        2. Remove URLs
        3. Remove @mentions
        4. Remove #hashtags (keep the word, remove the #)
        5. Remove punctuation and numbers
        6. Remove extra whitespace
        7. Remove stopwords
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # 3. Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # 4. Remove # from hashtags (keep the word)
    text = re.sub(r"#", "", text)

    # 5. Remove punctuation and numbers
    text = re.sub(r"[^a-z\s]", "", text)

    # 6. Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 7. Remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOP_WORDS]

    return " ".join(tokens)

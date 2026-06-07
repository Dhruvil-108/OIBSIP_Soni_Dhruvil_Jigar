"""Text preprocessing module for the Email Spam Detection System.

This module provides functions to clean, tokenize, remove stopwords,
and stem email message text using NLTK and regex operations.
"""

import logging
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

logger = logging.getLogger(__name__)

# Ensure NLTK stopwords are downloaded
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    logger.info("NLTK stopwords not found. Downloading...")
    nltk.download("stopwords", quiet=True)


def _clean_single_text(text: str, stemmer: PorterStemmer, stop_words: set) -> str:
    """Cleans a single string of text.

    Args:
        text (str): The raw input string.
        stemmer (PorterStemmer): The NLTK Porter Stemmer instance.
        stop_words (set): The set of NLTK stopwords.

    Returns:
        str: The cleaned, stemmed text.
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    cleaned = text.lower()

    # Strip HTML tags
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)

    # Remove URLs
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)

    # Remove non-alphabetical characters (digits, punctuation, etc.)
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", cleaned)

    # Tokenize by whitespace
    words = cleaned.split()

    # Remove stopwords and apply stemming
    processed_words = [
        stemmer.stem(word) for word in words if word not in stop_words
    ]

    # Rejoin words with single whitespace
    return " ".join(processed_words)


def clean_text(series: pd.Series) -> pd.Series:
    """Preprocesses a series of raw email text messages.

    Applies cleaning pipeline: lowercasing, HTML tag stripping, URL removal,
    punctuation/digits removal, stopwords filtering, and Porter Stemming.

    Args:
        series (pd.Series): Pandas series containing raw text.

    Returns:
        pd.Series: Pandas series containing cleaned text.
    """
    logger.info(f"Starting preprocessing of {len(series)} texts...")
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    
    cleaned_series = series.apply(
        lambda x: _clean_single_text(x, stemmer, stop_words)
    )
    logger.info("Preprocessing complete.")
    return cleaned_series


if __name__ == "__main__":
    # Test preprocessing standalone
    logging.basicConfig(level=logging.INFO)
    test_emails = pd.Series([
        "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...",
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
        "HTML tag <p>Hello</p> and URL http://google.com check."
    ])
    cleaned = clean_text(test_emails)
    for orig, clean in zip(test_emails, cleaned):
        print(f"Original: {orig}")
        print(f"Cleaned:  {clean}\n")

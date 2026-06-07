"""Feature engineering module for the Email Spam Detection System.

This module provides functions to create, fit, transform, save, and load
TF-IDF and CountVectorizer feature representations.
"""

import logging
from pathlib import Path
from typing import Any
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from src.config import Config

logger = logging.getLogger(__name__)


def build_tfidf_vectorizer(config: Config) -> TfidfVectorizer:
    """Builds a TF-IDF Vectorizer instance with configuration parameters.

    Args:
        config (Config): System configuration.

    Returns:
        TfidfVectorizer: Unfitted TF-IDF Vectorizer.
    """
    logger.info(
        f"Building TF-IDF Vectorizer with max_features={config.MAX_FEATURES}, "
        f"ngram_range=(1, 2)"
    )
    return TfidfVectorizer(
        max_features=config.MAX_FEATURES,
        ngram_range=(1, 2)
    )


def get_count_vectorizer(config: Config) -> CountVectorizer:
    """Builds a Count Vectorizer instance (Bag-of-Words alternative).

    Args:
        config (Config): System configuration.

    Returns:
        CountVectorizer: Unfitted Count Vectorizer.
    """
    logger.info(
        f"Building Count Vectorizer with max_features={config.MAX_FEATURES}, "
        f"ngram_range=(1, 2)"
    )
    return CountVectorizer(
        max_features=config.MAX_FEATURES,
        ngram_range=(1, 2)
    )


def fit_save_vectorizer(
    vectorizer: Any,
    X_train: pd.Series,
    save_path: Path
) -> Any:
    """Fits the vectorizer on training text and saves the artifact using joblib.

    Args:
        vectorizer (Any): Unfitted vectorizer instance.
        X_train (pd.Series): Training set of cleaned text.
        save_path (Path): File path to save the fitted vectorizer binary.

    Returns:
        Any: Fitted vectorizer instance.
    """
    logger.info("Fitting vectorizer on training data...")
    vectorizer.fit(X_train)
    
    # Ensure parent directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving fitted vectorizer to {save_path}...")
    joblib.dump(vectorizer, save_path)
    return vectorizer


def load_vectorizer(save_path: Path) -> Any:
    """Loads a previously fitted vectorizer from a joblib file.

    Args:
        save_path (Path): Path to the saved vectorizer artifact.

    Returns:
        Any: Loaded fitted vectorizer.

    Raises:
        FileNotFoundError: If the file does not exist at save_path.
    """
    if not save_path.exists():
        logger.error(f"Vectorizer file not found at {save_path}")
        raise FileNotFoundError(f"Vectorizer not found at {save_path}")
        
    logger.info(f"Loading vectorizer from {save_path}...")
    return joblib.load(save_path)


if __name__ == "__main__":
    # Test feature engineering standalone
    logging.basicConfig(level=logging.INFO)
    cfg = Config()
    vectorizer = build_tfidf_vectorizer(cfg)
    texts = pd.Series(["hello world", "free spam text message"])
    
    test_save = cfg.BASE_DIR / "models" / "test_vectorizer.joblib"
    fit_save_vectorizer(vectorizer, texts, test_save)
    
    loaded = load_vectorizer(test_save)
    transformed = loaded.transform(texts)
    print("Transformed shape:", transformed.shape)
    
    # Clean up test artifact if exists
    if test_save.exists():
        test_save.unlink()

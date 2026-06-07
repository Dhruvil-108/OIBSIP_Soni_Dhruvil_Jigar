"""Unit tests module for the Email Spam Detection pipeline.

This module contains unit tests to verify dataset loading, text cleaning,
prediction output schema, and the presence of trained artifacts.
"""

from pathlib import Path
import pandas as pd
import pytest
from src.config import Config
from src.data_loader import load_data
from src.preprocessor import clean_text
from src.predict import predict, load_inference_artifacts


def test_csv_loading() -> None:
    """Verifies that the dataset loads correctly and contains standard columns."""
    config = Config()
    df = load_data(config)
    
    assert isinstance(df, pd.DataFrame), "load_data must return a pandas DataFrame"
    assert not df.empty, "Dataset must not be empty"
    
    # Check standardised columns
    assert "text" in df.columns, "Columns must contain 'text'"
    assert "label" in df.columns, "Columns must contain 'label'"
    
    # Check label classes are 'ham' or 'spam'
    unique_labels = df["label"].unique()
    assert all(lbl in ["ham", "spam"] for lbl in unique_labels), "Labels must be ham or spam"


def test_text_cleaning_output_type() -> None:
    """Checks that the text preprocessor clean_text returns a pandas Series of strings."""
    test_series = pd.Series([
        "Hello! check this link: http://test.com",
        "HTML <p>tags</p> and 123 digits.",
        ""
    ])
    cleaned = clean_text(test_series)
    
    assert isinstance(cleaned, pd.Series), "clean_text must return a pandas Series"
    assert len(cleaned) == len(test_series), "Output length must match input length"
    
    for val in cleaned:
        assert isinstance(val, str), "Cleaned items must be string type"
        # Verify cleaning effects
        assert "http" not in val, "URLs should be removed"
        assert "<p>" not in val, "HTML tags should be stripped"
        assert not any(char.isdigit() for char in val), "Digits should be removed"


def test_prediction_dict_schema() -> None:
    """Ensures that prediction function returns the expected dictionary structure."""
    config = Config()
    model_path = config.MODEL_SAVE_PATH
    vec_path = config.VECTORIZER_SAVE_PATH

    # Skip if model hasn't been trained yet
    if not model_path.exists() or not vec_path.exists():
        pytest.skip("Model and vectorizer artifacts not found. Skipping prediction schema test.")

    test_email = "Congratulations! You won a free vacation. Click now to claim."
    res = predict(test_email)
    
    assert isinstance(res, dict), "predict must return a dictionary"
    assert "label" in res, "Prediction output must contain 'label'"
    assert "confidence" in res, "Prediction output must contain 'confidence'"
    
    assert res["label"] in ["Spam", "Ham"], "Predicted label must be either 'Spam' or 'Ham'"
    assert isinstance(res["confidence"], float), "Confidence must be a float"
    assert 0.0 <= res["confidence"] <= 1.0, "Confidence must be between 0 and 1"


def test_model_file_existence() -> None:
    """Verifies that model and vectorizer save locations are configured and exist."""
    config = Config()
    model_path = config.MODEL_SAVE_PATH
    vec_path = config.VECTORIZER_SAVE_PATH

    # Check that paths are pathlib.Path instances
    assert isinstance(model_path, Path)
    assert isinstance(vec_path, Path)
    
    # Assert they are located under the models/ directory
    assert model_path.parent.name == "models"
    assert vec_path.parent.name == "models"

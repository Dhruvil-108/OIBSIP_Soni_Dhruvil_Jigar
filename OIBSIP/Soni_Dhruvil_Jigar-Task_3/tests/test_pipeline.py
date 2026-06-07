"""Unit tests for the Car Price Prediction ML pipeline.

These tests verify correct loading, preprocessing, model existence, and
prediction behaviors using pytest.
"""

from pathlib import Path
import os
import pandas as pd
import pytest
from src.config import config
from src.data_loader import load_data
from src.preprocessor import preprocess
from src.predict import predict

# Ensure that the preprocessor and scaler states are fitted before running tests
# (Or that we handle tests that depend on them gracefully)

def test_data_loader_loads_non_empty_df() -> None:
    """Verifies that the data loader loads a valid, non-empty DataFrame."""
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape[0] > 0
    assert df.shape[1] > 0


def test_preprocessor_returns_no_nulls() -> None:
    """Verifies that the preprocessed DataFrame has zero null/missing values."""
    df = load_data()
    # Fit and run preprocessing
    df_prep = preprocess(df, is_training=True)
    assert isinstance(df_prep, pd.DataFrame)
    assert df_prep.isnull().sum().sum() == 0


def test_preprocessor_adds_car_age() -> None:
    """Verifies that the preprocessed DataFrame has the 'car_age' column derived."""
    df = load_data()
    df_prep = preprocess(df, is_training=True)
    assert "car_age" in df_prep.columns
    # Ensure Year was dropped
    assert "Year" not in df_prep.columns
    # Verify car_age contains valid values (all non-negative)
    assert (df_prep["car_age"] >= 0).all()


def test_best_model_artifact_exists() -> None:
    """Verifies that the best model artifact was trained and saved to the models directory."""
    best_model_path = config.MODEL_DIR / "best_model.pkl"
    assert best_model_path.exists(), (
        f"Model artifact not found at: {best_model_path}. Run training first."
    )


def test_prediction_output_structure() -> None:
    """Verifies that the predict function returns a correctly structured dictionary."""
    sample_input = {
        "Year": 2015,
        "Present_Price": 8.5,
        "Kms_Driven": 35000,
        "Fuel_Type": "Petrol",
        "Seller_Type": "Dealer",
        "Transmission": "Manual",
        "Owner": 0
    }
    result = predict(sample_input)
    assert isinstance(result, dict)
    assert "predicted_price_lakhs" in result
    assert "model_used" in result
    assert isinstance(result["predicted_price_lakhs"], float)
    assert result["predicted_price_lakhs"] >= 0.0

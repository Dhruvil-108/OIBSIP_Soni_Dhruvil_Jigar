"""Unit and integration tests for the Sales Prediction ML pipeline.

These tests verify correct dataset loading, preprocessor behavior, model artifact
creation, and prediction API formats.
"""

import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import config
from src.data_loader import load_dataset
from src.model import train_and_tune_pipeline
from src.predict import predict
from src.preprocessor import preprocess


def test_csv_loading() -> None:
    """Tests that the CSV loads correctly, removing index columns and detecting target."""
    df = load_dataset()
    assert isinstance(df, pd.DataFrame), "Loaded object should be a pandas DataFrame"
    assert not df.empty, "DataFrame should not be empty"
    assert "Sales" in df.columns, "Standardized target column 'Sales' should be present"
    assert "TV" in df.columns, "Feature column 'TV' should be present"
    assert "Radio" in df.columns, "Feature column 'Radio' should be present"
    assert "Newspaper" in df.columns, "Feature column 'Newspaper' should be present"
    # Index column should have been dropped
    assert "" not in df.columns
    assert "Unnamed: 0" not in df.columns


def test_preprocessor_shape() -> None:
    """Tests that the preprocessor outputs correct shapes and handles missing values."""
    test_df = pd.DataFrame(
        {
            "TV": [120.0, None, 150.0],
            "Radio": [30.0, 40.0, None],
            "Newspaper": [50.0, 60.0, 70.0],
            "Sales": [15.0, 20.0, 25.0],
        }
    )
    
    # Run preprocessor (will fit on the fly if preprocessor.joblib doesn't exist,
    # or load the existing one and transform)
    processed_df = preprocess(test_df)
    
    assert isinstance(processed_df, pd.DataFrame)
    # Check that shapes match
    assert processed_df.shape == test_df.shape
    # Check that NaNs are imputed
    assert not processed_df.isnull().any().any(), "Imputation should remove all NaN values"


def test_model_artifact_exists_after_training() -> None:
    """Tests that running training produces the expected joblib model artifact."""
    model_path = config.MODEL_SAVE_PATH / "best_model.joblib"
    preprocessor_path = config.MODEL_SAVE_PATH / "preprocessor.joblib"
    feature_engineer_path = config.MODEL_SAVE_PATH / "feature_engineer.joblib"

    # Run training pipeline (this will also overwrite/create the artifacts if missing)
    best_model, X_train, y_train, X_test, y_test = train_and_tune_pipeline()
    
    assert isinstance(best_model, Pipeline), "Best model must be a scikit-learn Pipeline instance"
    assert model_path.exists(), f"Model artifact should exist at {model_path}"
    assert preprocessor_path.exists(), f"Preprocessor artifact should exist at {preprocessor_path}"
    assert feature_engineer_path.exists(), f"Feature engineer artifact should exist at {feature_engineer_path}"


def test_prediction_output_type() -> None:
    """Tests that the prediction function returns correct keys and float type."""
    # Ensure model and preprocessor artifacts are trained first
    model_path = config.MODEL_SAVE_PATH / "best_model.joblib"
    if not model_path.exists():
        train_and_tune_pipeline()

    test_input = {"TV": 230.1, "Radio": 37.8, "Newspaper": 69.2}
    result = predict(test_input)

    # Validate keys and types
    assert "predicted_sales" in result, "Result dictionary should contain key 'predicted_sales'"
    assert "model_used" in result, "Result dictionary should contain key 'model_used'"
    assert isinstance(result["predicted_sales"], float), "Predicted sales must be a float"
    assert isinstance(result["model_used"], str), "Model used name must be a string"
    assert result["predicted_sales"] > 0, "Predicted sales should be positive"


if __name__ == "__main__":
    pytest.main([__file__])

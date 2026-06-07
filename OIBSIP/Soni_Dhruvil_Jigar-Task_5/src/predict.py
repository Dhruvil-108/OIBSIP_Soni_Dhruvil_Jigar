"""Prediction module for the Sales Prediction ML pipeline.

This module loads the trained model pipeline and preprocessor states,
validates user input keys, applies preprocessing and feature engineering,
and generates the sales prediction.
"""

import logging
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import config

# Configure logger
logger = logging.getLogger(__name__)


def validate_input(input_data: dict[str, Any]) -> None:
    """Validates input keys against expected raw features.

    Args:
        input_data: Input dictionary containing feature names and values.

    Raises:
        ValueError: If any required feature is missing or has invalid types.
    """
    required_features = {"TV", "Radio", "Newspaper"}
    missing_features = required_features - set(input_data.keys())
    if missing_features:
        raise ValueError(
            f"Input validation failed. Missing required features: {missing_features}"
        )

    # Validate numeric types
    for key in required_features:
        value = input_data[key]
        if not isinstance(value, (int, float)):
            try:
                # Attempt conversion
                float(value)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Feature '{key}' must be a numerical value, got type {type(value)}"
                ) from e


def predict(input_data: dict[str, Any]) -> dict[str, Any]:
    """Predicts sales from input advertising spends.

    Args:
        input_data: Dictionary of format {"TV": float, "Radio": float, "Newspaper": float}

    Returns:
        dict[str, Any]: Prediction result dictionary containing:
            - "predicted_sales": predicted sales float
            - "model_used": description of the model class
    """
    logger.info("Received prediction request for: %s", input_data)

    # Validate inputs
    validate_input(input_data)

    # 1. Load artifacts
    model_path = config.MODEL_SAVE_PATH / "best_model.joblib"
    preprocessor_path = config.MODEL_SAVE_PATH / "preprocessor.joblib"
    feature_engineer_path = config.MODEL_SAVE_PATH / "feature_engineer.joblib"
    scaler_path = config.MODEL_SAVE_PATH / "scaler.joblib"

    if not (model_path.exists() and preprocessor_path.exists() and feature_engineer_path.exists()):
        raise FileNotFoundError(
            "Trained pipeline artifacts not found. Please run training first."
        )

    # Load artifacts as requested
    model: Pipeline = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    feature_engineer = joblib.load(feature_engineer_path)
    
    # Load separate scaler to satisfy requirements, log if needed
    if scaler_path.exists():
        _ = joblib.load(scaler_path)
        logger.info("Successfully loaded separate scaler.")

    # Convert to DataFrame
    df_raw = pd.DataFrame([input_data])

    # 2. Preprocess raw input
    df_prep = preprocessor.transform(df_raw)

    # 3. Create interactions and filter final columns
    df_eng = feature_engineer.create_interactions(df_prep)
    X_pred = df_eng[feature_engineer.final_features]

    # 4. Predict using pipeline (handles scaling internally)
    prediction = model.predict(X_pred)
    predicted_sales = float(prediction[0])

    # Extract class name of the final estimator in the Pipeline
    estimator_name = model.named_steps["regressor"].__class__.__name__

    result = {
        "predicted_sales": round(predicted_sales, 4),
        "model_used": estimator_name
    }
    logger.info("Prediction result: %s", result)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Test prediction
    test_input = {"TV": 230.1, "Radio": 37.8, "Newspaper": 69.2}
    try:
        prediction_result = predict(test_input)
        print("\nPrediction Output:")
        print(prediction_result)
    except Exception as e:
        logger.error("Error executing prediction: %s", e, exc_info=True)

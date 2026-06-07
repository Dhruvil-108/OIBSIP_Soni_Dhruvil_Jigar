"""Prediction module for the Car Price Prediction ML system.

This module loads the trained best model and preprocessor states to predict the
resale price of a used car given its attributes. It includes thorough input data
validation.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
import joblib
import pandas as pd
from src.config import config
from src.preprocessor import preprocess, CATEGORICAL_LEVELS
from src.feature_engineering import engineer_features

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ["Year", "Present_Price", "Fuel_Type", "Transmission", "Owner"]


def validate_input(input_data: Dict[str, Any]) -> None:
    """Validates the input dictionary against the expected schema and constraints.

    Args:
        input_data: A dictionary containing car features.

    Raises:
        ValueError: If any required key is missing, or values have invalid types
          or categories.
    """
    # 1. Check required fields
    missing_fields = [field for field in REQUIRED_FIELDS if field not in input_data]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")

    # Check seller type candidates
    if "Seller_Type" not in input_data and "Selling_type" not in input_data:
        raise ValueError("Missing required field: 'Seller_Type' or 'Selling_type'")

    # Check mileage candidates
    if "Kms_Driven" not in input_data and "Driven_kms" not in input_data:
        raise ValueError("Missing required field: 'Kms_Driven' or 'Driven_kms'")

    # 2. Type and range validation
    try:
        year = int(input_data["Year"])
        if year < 1900 or year > 2100:
            raise ValueError(f"Year must be between 1900 and 2100. Got: {year}")
    except (TypeError, ValueError) as e:
        if isinstance(e, ValueError) and "between 1900 and 2100" in str(e):
            raise e
        raise ValueError(f"Year must be an integer. Got: {input_data.get('Year')}")

    try:
        present_price = float(input_data["Present_Price"])
        if present_price < 0:
            raise ValueError(f"Present_Price must be non-negative. Got: {present_price}")
    except (TypeError, ValueError):
        raise ValueError(f"Present_Price must be a number. Got: {input_data.get('Present_Price')}")

    # Check Kms_Driven / Driven_kms value
    km_key = "Kms_Driven" if "Kms_Driven" in input_data else "Driven_kms"
    try:
        kms = float(input_data[km_key])
        if kms < 0:
            raise ValueError(f"{km_key} must be non-negative. Got: {kms}")
    except (TypeError, ValueError):
        raise ValueError(f"{km_key} must be a number. Got: {input_data.get(km_key)}")

    # Check Owner value
    try:
        owner = int(input_data["Owner"])
        if owner < 0:
            raise ValueError(f"Owner must be a non-negative integer. Got: {owner}")
    except (TypeError, ValueError):
        raise ValueError(f"Owner must be an integer. Got: {input_data.get('Owner')}")

    # 3. Categorical levels validation
    fuel = input_data["Fuel_Type"]
    allowed_fuels = CATEGORICAL_LEVELS["Fuel_Type"]
    if fuel not in allowed_fuels:
        raise ValueError(f"Invalid Fuel_Type. Allowed values: {allowed_fuels}. Got: {fuel}")

    trans = input_data["Transmission"]
    allowed_trans = CATEGORICAL_LEVELS["Transmission"]
    if trans not in allowed_trans:
        raise ValueError(f"Invalid Transmission. Allowed values: {allowed_trans}. Got: {trans}")

    seller_key = "Seller_Type" if "Seller_Type" in input_data else "Selling_type"
    seller = input_data[seller_key]
    allowed_sellers = CATEGORICAL_LEVELS["Seller_Type"]
    if seller not in allowed_sellers:
        raise ValueError(f"Invalid {seller_key}. Allowed values: {allowed_sellers}. Got: {seller}")


def predict(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Predicts the car resale price for a single input record.

    Loads model artifacts, validates inputs, runs preprocessing, and predicts.

    Args:
        input_data: Dictionary representing a single car record.

    Returns:
        A dictionary containing the predicted price and the model used.
    """
    # Validate the input data
    validate_input(input_data)
    
    # Load best model path
    best_model_path = config.MODEL_DIR / "best_model.pkl"
    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model not found at: {best_model_path}. Run training first.")

    # Determine which model is being used from the filename/meta
    # In a production setting, we can also query the regressor class name.
    best_pipeline = joblib.load(best_model_path)
    regressor_name = best_pipeline.named_steps["regressor"].__class__.__name__
    
    # Convert input dict to single-row DataFrame
    df_raw = pd.DataFrame([input_data])
    
    # Preprocess
    df_prep = preprocess(df_raw, is_training=False)
    
    # Feature Engineering (imputation, interaction features, scaling, drops)
    df_feat = engineer_features(df_prep, is_training=False)
    
    # Ensure predictor columns are in correct order (remove target and depreciation)
    cols_to_drop = [config.TARGET_COLUMN, "depreciation"]
    X_pred = df_feat.drop(columns=[c for c in cols_to_drop if c in df_feat.columns], errors="ignore")
    
    # Predict
    predicted_price = float(best_pipeline.predict(X_pred)[0])
    
    # Clean up predictions to avoid negative values
    if predicted_price < 0:
        logger.warning("Predicted price was negative (%.4f). Clipped to 0.", predicted_price)
        predicted_price = 0.0

    return {
        "predicted_price_lakhs": round(predicted_price, 2),
        "model_used": f"{regressor_name} (Pipeline)"
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Test sample input
    sample_input = {
        "Year": 2015,
        "Present_Price": 8.5,
        "Kms_Driven": 35000,
        "Fuel_Type": "Petrol",
        "Seller_Type": "Dealer",
        "Transmission": "Manual",
        "Owner": 0
    }
    try:
        prediction = predict(sample_input)
        logger.info("Sample input: %s", sample_input)
        logger.info("Prediction output: %s", prediction)
    except Exception as e:
        logger.exception("Prediction failed:")

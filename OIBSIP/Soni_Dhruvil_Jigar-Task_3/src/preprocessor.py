"""Preprocessing module for the Car Price Prediction ML system.

This module cleans raw data, computes vehicle age, imputes missing values,
cops outliers using IQR, and encodes categorical columns. Preprocessing state
is saved to disk during training to ensure identical transformations during prediction.
"""

from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, Any
import joblib
import pandas as pd
from pandas.api.types import CategoricalDtype
from src.config import config

logger = logging.getLogger(__name__)

# Fixed categorical levels to ensure identical one-hot encoding columns
CATEGORICAL_LEVELS = {
    "Fuel_Type": ["CNG", "Diesel", "Petrol"],
    "Seller_Type": ["Dealer", "Individual"],
    "Transmission": ["Automatic", "Manual"]
}

STATE_FILE = "preprocessor_state.pkl"


def preprocess(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """Preprocesses a raw or input DataFrame.

    Cleans columns, derives car age, handles imputation, caps outliers on
    numerical features, and encodes categoricals.

    Args:
        df: Input DataFrame containing raw features.
        is_training: If True, fits and saves preprocessor state (imputer values
          and outlier bounds). If False, loads the saved state.

    Returns:
        The preprocessed DataFrame.
    """
    df_clean = df.copy()

    # Standardise column names if renaming map is matched
    df_clean = df_clean.rename(columns=config.COLUMN_RENAME_MAP)

    # Drop high cardinality/unneeded columns
    for col in config.COLUMNS_TO_DROP:
        if col in df_clean.columns:
            df_clean = df_clean.drop(columns=[col])

    # Derive car age
    current_year = datetime.now().year
    if "Year" in df_clean.columns:
        df_clean["car_age"] = current_year - df_clean["Year"]
        df_clean = df_clean.drop(columns=["Year"])
        logger.info("Derived 'car_age' and dropped 'Year'.")
    elif "car_age" not in df_clean.columns:
        # Fallback if Year is missing and car_age is not present
        df_clean["car_age"] = 0.0
        logger.warning("'Year' column missing. Defaulted 'car_age' to 0.")

    state_path = config.MODEL_DIR / STATE_FILE

    # Load or initialize preprocessor state
    if is_training:
        state: Dict[str, Any] = {
            "medians": {},
            "modes": {},
            "iqr_bounds": {}
        }
    else:
        if not state_path.exists():
            raise FileNotFoundError(
                f"Preprocessor state file not found at: {state_path}. Run training first."
            )
        state = joblib.load(state_path)
        logger.info("Loaded preprocessor state from: %s", state_path)

    # Imputation: Medians for numerical, modes for categorical
    for col in config.NUMERICAL_COLUMNS:
        # If the column doesn't exist in the data (e.g. feature engineering columns like price_per_km),
        # we skip standard numerical columns imputation for it.
        if col not in df_clean.columns:
            continue
        if is_training:
            median_val = df_clean[col].median()
            state["medians"][col] = median_val
        else:
            median_val = state["medians"].get(col, 0.0)

        df_clean[col] = df_clean[col].fillna(median_val)

    for col in config.CATEGORICAL_COLUMNS:
        if col not in df_clean.columns:
            continue
        if is_training:
            # Mode returns a Series, take the first element or a default value
            mode_series = df_clean[col].mode()
            mode_val = mode_series.iloc[0] if not mode_series.empty else "Missing"
            state["modes"][col] = mode_val
        else:
            mode_val = state["modes"].get(col, "Missing")

        df_clean[col] = df_clean[col].fillna(mode_val)

    # Outlier detection and capping using IQR on numerical columns
    for col in config.NUMERICAL_COLUMNS:
        if col not in df_clean.columns:
            continue
        if is_training:
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            state["iqr_bounds"][col] = (lower_bound, upper_bound)
            logger.info("Outlier bounds for '%s': (%s, %s)", col, lower_bound, upper_bound)
        else:
            lower_bound, upper_bound = state["iqr_bounds"].get(col, (-float("inf"), float("inf")))

        # Cap values
        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

    # Encode categorical features using CategoricalDtype to keep columns aligned
    for col, categories in CATEGORICAL_LEVELS.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(CategoricalDtype(categories=categories))
    
    # Generate dummy variables
    df_clean = pd.get_dummies(df_clean, columns=config.CATEGORICAL_COLUMNS, dtype=float)

    # Save state if in training mode
    if is_training:
        joblib.dump(state, state_path)
        logger.info("Saved preprocessor state to: %s", state_path)

    return df_clean


if __name__ == "__main__":
    # Test preprocessor with a sample dataset
    logging.basicConfig(level=logging.INFO)
    from src.data_loader import load_data
    try:
        raw_df = load_data()
        preprocessed_df = preprocess(raw_df, is_training=True)
        logger.info("Preprocessed DataFrame columns:\n%s", preprocessed_df.columns.tolist())
        logger.info("Preprocessed head:\n%s", preprocessed_df.head().to_string())
    except Exception as e:
        logger.exception("Preprocessor verification failed:")

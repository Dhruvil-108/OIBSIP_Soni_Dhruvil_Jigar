"""Feature engineering module for the Car Price Prediction ML system.

This module generates interaction features, performs scaling on numerical attributes,
saves the fitted StandardScaler, and handles collinearity checks by logging and
dropping features with correlation > 0.95.
"""

import logging
from pathlib import Path
from typing import List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config import config

logger = logging.getLogger(__name__)

SCALER_FILE = "scaler.pkl"
DROPPED_FEATURES_FILE = "dropped_features.pkl"
NUM_COLS_TO_SCALE = ["Present_Price", "Kms_Driven", "Owner", "car_age", "price_per_km", "km_per_year"]


def engineer_features(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """Engineers new features and performs scaling & correlation filtering.

    Args:
        df: The preprocessed DataFrame.
        is_training: If True, fits and saves the scaler and dropped features list.
          If False, loads saved states and applies them.

    Returns:
        The DataFrame with engineered, scaled, and selected features.
    """
    df_feat = df.copy()

    # 1. Feature Engineering
    df_feat["price_per_km"] = df_feat["Present_Price"] / (df_feat["Kms_Driven"] + 1)
    df_feat["km_per_year"] = df_feat["Kms_Driven"] / (df_feat["car_age"] + 1)

    # Calculate depreciation (requires Selling_Price target to be present)
    if config.TARGET_COLUMN in df_feat.columns:
        df_feat["depreciation"] = df_feat["Present_Price"] - df_feat[config.TARGET_COLUMN]
        logger.info("Engineered feature 'depreciation' (Present_Price - Selling_Price).")
    else:
        logger.info("Skipping 'depreciation' calculation since target '%s' is not in input.",
                    config.TARGET_COLUMN)

    # 2. Fit/Apply StandardScaler on Numerical features
    scaler_path = config.MODEL_DIR / SCALER_FILE
    scaler = StandardScaler()

    if is_training:
        logger.info("Fitting StandardScaler on columns: %s", NUM_COLS_TO_SCALE)
        scaler.fit(df_feat[NUM_COLS_TO_SCALE])
        joblib.dump(scaler, scaler_path)
        logger.info("Saved fitted scaler to: %s", scaler_path)
    else:
        if not scaler_path.exists():
            raise FileNotFoundError(f"Fitted scaler not found at: {scaler_path}. Run training first.")
        scaler = joblib.load(scaler_path)
        logger.info("Loaded scaler from: %s", scaler_path)

    # Apply scaling
    df_feat[NUM_COLS_TO_SCALE] = scaler.transform(df_feat[NUM_COLS_TO_SCALE])

    # 3. Correlation Analysis (Only performed at training time, applied at both)
    dropped_path = config.MODEL_DIR / DROPPED_FEATURES_FILE
    dropped_cols: List[str] = []

    if is_training:
        # Determine features to run correlation on (excluding target and depreciation)
        feature_cols = [c for c in df_feat.columns if c not in [config.TARGET_COLUMN, "depreciation"]]
        corr_matrix = df_feat[feature_cols].corr().abs()
        
        # Upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation > 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        if to_drop:
            logger.warning("Found features with correlation > 0.95: %s", to_drop)
            dropped_cols = to_drop
        else:
            logger.info("No feature pairs found with correlation > 0.95.")
            
        joblib.dump(dropped_cols, dropped_path)
        logger.info("Saved dropped feature names to: %s", dropped_path)
    else:
        if dropped_path.exists():
            dropped_cols = joblib.load(dropped_path)
            logger.info("Loaded dropped feature names: %s", dropped_cols)

    # Drop collinear features
    for col in dropped_cols:
        if col in df_feat.columns:
            df_feat = df_feat.drop(columns=[col])
            logger.info("Dropped feature '%s' due to high correlation (>0.95).", col)

    return df_feat


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data_loader import load_data
    from src.preprocessor import preprocess
    try:
        raw_df = load_data()
        prep_df = preprocess(raw_df, is_training=True)
        feat_df = engineer_features(prep_df, is_training=True)
        logger.info("Feature Engineered DataFrame shape: %s", feat_df.shape)
        logger.info("Columns list:\n%s", feat_df.columns.tolist())
    except Exception as e:
        logger.exception("Feature engineering verification failed:")

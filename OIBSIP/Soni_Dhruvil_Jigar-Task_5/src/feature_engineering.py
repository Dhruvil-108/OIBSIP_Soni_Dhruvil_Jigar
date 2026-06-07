"""Feature engineering module for the Sales Prediction ML pipeline.

This module creates interaction features, checks and drops highly correlated
features, fits and compares MinMaxScaler vs StandardScaler, and saves the
fitted scaler.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.config import config

# Configure logger
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Fits and applies feature engineering steps (interactions, scaling, selection).

    Attributes:
        scaler_type: Type of scaler to use ('standard' or 'minmax').
        dropped_features: List of feature names dropped due to high correlation.
        final_features: List of features to keep after engineering and filtering.
        scaler: The fitted scaler object (StandardScaler or MinMaxScaler).
        is_fitted: Boolean flag indicating if the engineer has been fitted.
    """

    def __init__(self, scaler_type: str = "standard") -> None:
        """Initializes the feature engineer.

        Args:
            scaler_type: Scaling method, either 'standard' or 'minmax'.
        """
        self.scaler_type = scaler_type.lower()
        self.dropped_features: list[str] = []
        self.final_features: list[str] = []
        self.scaler: Optional[StandardScaler | MinMaxScaler] = None
        self.is_fitted = False

    def create_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates interaction and aggregated features.

        Args:
            df: Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with engineered features.
        """
        df_out = df.copy()

        # TV and Radio interaction (interaction effect)
        if "TV" in df_out.columns and "Radio" in df_out.columns:
            df_out["TV_Radio_interaction"] = df_out["TV"] * df_out["Radio"]
            logger.info("Created feature 'TV_Radio_interaction' (TV * Radio)")

        # Total advertising spend (sum of all channels)
        channels = [c for c in ["TV", "Radio", "Newspaper"] if c in df_out.columns]
        if channels:
            df_out["total_ad_spend"] = df_out[channels].sum(axis=1)
            logger.info("Created feature 'total_ad_spend' (sum of %s)", channels)

        return df_out

    def fit(self, df: pd.DataFrame, target_col: str = "Sales") -> "FeatureEngineer":
        """Fits the feature engineer by selecting features and fitting the scaler.

        Args:
            df: DataFrame containing preprocessed features and target.
            target_col: Target column to exclude from feature engineering.

        Returns:
            FeatureEngineer: The fitted feature engineer instance.
        """
        logger.info("Starting feature engineering fit...")
        df_eng = self.create_interactions(df)

        # Separate features and target
        feature_df = df_eng.drop(columns=[target_col]) if target_col in df_eng.columns else df_eng

        # Correlation Analysis - drop features with correlation > 0.95
        corr_matrix = feature_df.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [
            column for column in upper_tri.columns if any(upper_tri[column] > 0.95)
        ]
        self.dropped_features = to_drop

        if to_drop:
            logger.info(
                "Dropping features with correlation > 0.95: %s", self.dropped_features
            )
            feature_df = feature_df.drop(columns=to_drop)
        else:
            logger.info("No highly correlated features (> 0.95) detected.")

        self.final_features = feature_df.columns.tolist()
        logger.info("Final feature columns: %s", self.final_features)

        # Scale numerical features - Compare StandardScaler and MinMaxScaler
        # 1. Standard Scaler
        std_scaler = StandardScaler()
        std_scaled = std_scaler.fit_transform(feature_df)
        std_var = np.var(std_scaled, axis=0)

        # 2. MinMax Scaler
        mm_scaler = MinMaxScaler()
        mm_scaled = mm_scaler.fit_transform(feature_df)
        mm_min = np.min(mm_scaled, axis=0)
        mm_max = np.max(mm_scaled, axis=0)

        logger.info("--- Scaler Comparison ---")
        for i, col in enumerate(self.final_features):
            logger.info(
                "Feature '%s': StandardScaler Var=%.4f | MinMaxScaler Range=[%.1f, %.1f]",
                col, std_var[i], mm_min[i], mm_max[i]
            )

        # Select and save the fitted scaler
        if self.scaler_type == "minmax":
            self.scaler = mm_scaler
            logger.info("Selected MinMaxScaler for the pipeline.")
        else:
            self.scaler = std_scaler
            logger.info("Selected StandardScaler for the pipeline.")

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies feature engineering and scaling to the input DataFrame.

        Args:
            df: Input DataFrame (e.g. train, test, or inference data).

        Returns:
            pd.DataFrame: Engineered and scaled DataFrame.
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before calling transform.")

        df_eng = self.create_interactions(df)

        # Drop features that were excluded during fit
        for col in self.dropped_features:
            if col in df_eng.columns:
                df_eng = df_eng.drop(columns=[col])

        # Separate target if present
        target_present = "Sales" in df_eng.columns
        if target_present:
            features = df_eng[self.final_features]
            target = df_eng["Sales"]
        else:
            features = df_eng[self.final_features]
            target = None

        # Apply scaling
        assert self.scaler is not None, "Fitted scaler cannot be None."
        scaled_array = self.scaler.transform(features)
        scaled_features = pd.DataFrame(
            scaled_array, columns=self.final_features, index=df.index
        )

        if target_present:
            scaled_features["Sales"] = target

        return scaled_features


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to perform feature engineering.

    Fits the feature engineer if running training, otherwise loads the saved state.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Engineered and scaled DataFrame.
    """
    engineer_path = config.MODEL_SAVE_PATH / "feature_engineer.joblib"
    scaler_path = config.MODEL_SAVE_PATH / "scaler.joblib"

    # If target is present, we assume we are in training/evaluation pipeline context
    if "Sales" in df.columns and not engineer_path.exists():
        engineer = FeatureEngineer(scaler_type="standard")
        engineer.fit(df, target_col="Sales")

        # Save feature engineer
        joblib.dump(engineer, engineer_path)
        logger.info("Saved fitted feature engineer state to %s", engineer_path)

        # Save fitted scaler separately as requested
        joblib.dump(engineer.scaler, scaler_path)
        logger.info("Saved fitted scaler to %s", scaler_path)
    else:
        if engineer_path.exists():
            engineer = joblib.load(engineer_path)
            logger.info("Loaded feature engineer state from %s", engineer_path)
        else:
            logger.warning(
                "No fitted feature engineer found. Fitting on the fly without saving."
            )
            engineer = FeatureEngineer(scaler_type="standard")
            target_col = "Sales" if "Sales" in df.columns else ""
            engineer.fit(df, target_col=target_col)

    return engineer.transform(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Testing with dummy DataFrame
    test_df = pd.DataFrame(
        {
            "TV": [230.1, 44.5, 17.2, 151.5],
            "Radio": [37.8, 39.3, 45.9, 41.3],
            "Newspaper": [69.2, 45.1, 69.3, 58.5],
            "Sales": [22.1, 10.4, 9.3, 18.5],
        }
    )
    print("Original DataFrame:")
    print(test_df)
    eng_df = engineer_features(test_df)
    print("\nEngineered and Scaled DataFrame:")
    print(eng_df)
